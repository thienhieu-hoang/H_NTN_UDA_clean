"""
HA02 Channel Estimation Model Architecture (TensorFlow / Keras)
===============================================================
Converted from PyTorch to TensorFlow 2.x.

Architecture Summary:
- Input:  (Batch, 88, 2)       --> Sparse LS estimates at pilot locations (Real & Imag)
- Output: (Batch, 14, 132, 2)   --> Reconstructed 2D channel grid (14 symbols x 132 subcarriers x 2)

Sub-modules:
1. TransformerEncoderBlock: Multi-Head Self-Attention on pilot features + Feed-Forward Network.
2. ResidualConvDecoderBlock: Conv2D -> Residual Block (Conv-ReLU-Conv+BatchNorm) -> Linear Upsampler (88 -> 1848) -> Conv2D.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses


# =============================================================================
# 1. TRANSFORMER ENCODER BLOCK (Attention Pre-processor)
# =============================================================================

class TransformerEncoderBlock(layers.Layer):
    """
    Transformer Encoder Block for HA02 (Section IV-A in the paper).
    - Multi-Head Self-Attention on pilot features (N_heads = N_pilot = 2).
    - Add & Layer Normalization.
    - Feed-Forward Network (FC -> GeLU -> FC).
    - Add & Layer Normalization.
    """
    def __init__(self, num_pilot_elems=88, num_channels=2, num_heads=2, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.num_pilot_elems = num_pilot_elems  # 88
        self.num_channels = num_channels        # 2 (Real, Imag)
        self.num_heads = num_heads              # 2 heads
        
        self.in_dim = num_pilot_elems * num_channels  # 88 * 2 = 176
        self.head_dim = self.in_dim // num_heads      # 176 / 2 = 88
        
        # FC1: resizes input from (176) to 3 * 176 = 528 for Q, K, V projection
        self.fc1 = layers.Dense(3 * self.in_dim, name="qkv_projection")
        
        # FC2: projects concatenated multi-head attention back to in_dim
        self.fc2 = layers.Dense(self.in_dim, name="attn_out_projection")
        
        # Layer Normalization 1 & 2
        self.ln1 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm_1")
        self.ln2 = layers.LayerNormalization(epsilon=1e-5, name="layer_norm_2")
        
        # Feed-Forward Network: FC -> GeLU -> FC
        self.ffn_dense1 = layers.Dense(self.in_dim * 2, name="ffn_dense1")
        self.ffn_dense2 = layers.Dense(self.in_dim, name="ffn_dense2")

    def call(self, inputs):
        """
        Input shape: (batch_size, num_pilot_elems, num_channels) e.g., (B, 88, 2)
        """
        B = tf.shape(inputs)[0]
        x_flat = tf.reshape(inputs, [B, self.in_dim])  # (B, 176)
        
        # 1. Linear projection to generate Q, K, V
        qkv = self.fc1(x_flat)  # (B, 528)
        qkv = tf.reshape(qkv, [B, 3, self.num_heads, self.head_dim])  # (B, 3, 2, 88)
        Q = qkv[:, 0, :, :]  # (B, 2, 88)
        K = qkv[:, 1, :, :]  # (B, 2, 88)
        V = qkv[:, 2, :, :]  # (B, 2, 88)
        
        # 2. Scaled Dot-Product Attention per head
        # Scale factor: sqrt(num_pilot_elems / num_heads) = sqrt(88 / 2) = sqrt(44) ≈ 6.633
        scale = tf.cast(tf.sqrt(self.num_pilot_elems / self.num_heads), dtype=tf.float32)
        
        Q_exp = tf.expand_dims(Q, axis=-1)  # (B, 2, 88, 1)
        K_exp = tf.expand_dims(K, axis=-2)  # (B, 2, 1, 88)
        
        scores = tf.matmul(Q_exp, K_exp) / scale  # (B, 2, 88, 88)
        attn_weights = tf.nn.softmax(scores, axis=-1)
        
        V_exp = tf.expand_dims(V, axis=-1)  # (B, 2, 88, 1)
        attn_out = tf.squeeze(tf.matmul(attn_weights, V_exp), axis=-1)  # (B, 2, 88)
        
        # 3. Concatenate heads & project
        attn_out_flat = tf.reshape(attn_out, [B, self.in_dim])  # (B, 176)
        attn_proj = self.fc2(attn_out_flat)                    # (B, 176)
        
        # 4. Residual + Add & Norm 1
        x_norm1 = self.ln1(x_flat + attn_proj)
        
        # 5. Feed-Forward Network + Add & Norm 2
        ffn1 = tf.nn.gelu(self.ffn_dense1(x_norm1))
        ffn_out = self.ffn_dense2(ffn1)
        out = self.ln2(x_norm1 + ffn_out)  # (B, 176)
        
        # Reshape back to (B, 88, 2)
        return tf.reshape(out, [B, self.num_pilot_elems, self.num_channels])


# =============================================================================
# 2. RESIDUAL CONVOLUTIONAL DECODER BLOCK (Decoder + Upsampler)
# =============================================================================

class ResidualConvDecoderBlock(layers.Layer):
    """
    Residual Convolutional Architecture for HA02 (Section IV-B in the paper).
    - Conv2D (2x2 kernel, N_filter=2 channels)
    - Residual Module (Conv2D -> ReLU -> Conv2D + BatchNorm)
    - Upsampling Module (Dense layer projecting 88 pilot dimension -> 1848 full grid dimension)
    - Conv2D output layer (maps 2 filters back to 1 channel)
    - Reshape to 2D channel grid (14 symbols x 132 subcarriers x 2)
    """
    def __init__(self, num_pilot_elems=88, total_grid_elems=1848, n_filter=2, **kwargs):
        super(ResidualConvDecoderBlock, self).__init__(**kwargs)
        self.num_pilot_elems = num_pilot_elems    # 88
        self.total_grid_elems = total_grid_elems  # N_s * N_f = 14 * 132 = 1848
        self.num_subcarriers = total_grid_elems // 14
        self.n_filter = n_filter                  # 2 filters
        
        # First Conv Layer (operates on 2D feature map [88, 2, 1])
        self.conv1 = layers.Conv2D(filters=n_filter, kernel_size=(2, 2), padding='same', name="conv1")
        
        # Residual Block: Conv -> ReLU -> Conv
        self.res_conv1 = layers.Conv2D(filters=n_filter, kernel_size=(2, 2), padding='same', name="res_conv1")
        self.relu = layers.ReLU()
        self.res_conv2 = layers.Conv2D(filters=n_filter, kernel_size=(2, 2), padding='same', name="res_conv2")
        self.norm = layers.BatchNormalization(name="batch_norm")
        
        # FC Layer for Upsampling from pilot dimension (88) to full slot dimension (1848 = 14 * 132)
        self.fc_upsample = layers.Dense(total_grid_elems, name="fc_upsample")
        
        # Final Convolutional layer (maps n_filter -> 1 channel)
        self.conv_out = layers.Conv2D(filters=1, kernel_size=(2, 2), padding='same', name="conv_out")

    def call(self, inputs, training=False):
        """
        Input shape: (B, num_pilot_elems, 2) from Transformer Encoder
        """
        B = tf.shape(inputs)[0]
        
        # Format as 2D spatial feature map: (B, self.num_pilot_elems, 2, 1) [NHWC]
        x_img = tf.expand_dims(inputs, axis=-1)
        
        # Conv 1
        h1 = self.conv1(x_img)  # (B, self.num_pilot_elems, 2, n_filter)
        
        # Residual Block
        res = self.res_conv1(h1)
        res = self.relu(res)
        res = self.res_conv2(res)
        h2 = self.norm(h1 + res, training=training)  # (B, self.num_pilot_elems, 2, n_filter)
        
        # Upsampling via Dense layer along spatial pilot axis (H=num_pilot_elems -> total_grid_elems)
        # Transpose H (num_pilot_elems) to last dimension for Dense projection: (B, n_filter, 2, num_pilot_elems)
        h2_trans = tf.transpose(h2, [0, 3, 2, 1])
        h2_upsampled = self.fc_upsample(h2_trans)   # (B, n_filter, 2, self.total_grid_elems)
        h2_upsampled = tf.transpose(h2_upsampled, [0, 3, 2, 1])  # (B, self.total_grid_elems, 2, n_filter)
        
        # Final Conv layer
        out = self.conv_out(h2_upsampled)  # (B, self.total_grid_elems, 2, 1)
        out = tf.squeeze(out, axis=-1)     # (B, self.total_grid_elems, 2)
        
        # Reshape to 2D channel grid (B, 14, num_subcarriers, 2) corresponding to (N_s, N_f, Real/Imag)
        out_grid = tf.reshape(out, [B, 14, self.num_subcarriers, 2])
        return out_grid


# =============================================================================
# 3. COMPLETE HA02 MODEL (Keras Model)
# =============================================================================

class HA02Model(models.Model):
    """
    Complete HA02 Hybrid Architecture combining:
    1. Transformer Encoder Stack (Attention pre-processor for sparse LS pilots)
    2. Residual Convolutional Architecture (Decoder + Upsampler to full 14x132 grid)
    """
    def __init__(self, num_pilot_elems=88, total_grid_elems=1848, num_channels=2, num_heads=2, n_filter=2, **kwargs):
        super(HA02Model, self).__init__(**kwargs)
        self.encoder = TransformerEncoderBlock(
            num_pilot_elems=num_pilot_elems, 
            num_channels=num_channels, 
            num_heads=num_heads
        )
        self.decoder = ResidualConvDecoderBlock(
            num_pilot_elems=num_pilot_elems, 
            total_grid_elems=total_grid_elems, 
            n_filter=n_filter
        )

    def call(self, inputs, training=False):
        """
        Input:  (batch_size, num_pilot_elems, 2)     -- Sparse LS estimates at pilot locations (Real & Imag)
        Output: (batch_size, 14, num_subcarriers, 2) -- Reconstructed full channel grid (14 symbols x num_subcarriers x 2)
        """
        encoder_out = self.encoder(inputs)
        full_grid_out = self.decoder(encoder_out, training=training)
        return full_grid_out


# =============================================================================
# 4. HUBER LOSS FUNCTION
# =============================================================================

class HuberLoss(losses.Loss):
    """
    Huber Loss with transition threshold delta = 1.0 (Equation 8 in the paper).
    Less sensitive to outliers than MSE.
    """
    def __init__(self, delta=1.0, name="huber_loss", **kwargs):
        super(HuberLoss, self).__init__(name=name, **kwargs)
        self.delta = delta

    def call(self, y_true, y_pred):
        err = tf.abs(y_pred - y_true)
        huber_err = tf.where(
            err <= self.delta,
            0.5 * tf.square(err),
            self.delta * (err - 0.5 * self.delta)
        )
        return tf.reduce_mean(huber_err)


# =============================================================================
# 5. MAIN TEST & SUMMARY SCRIPT
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("      HA02 CHANNEL ESTIMATION MODEL IMPLEMENTATION (TENSORFLOW)       ")
    print("=" * 70)
    
    # 1. Instantiate Model
    model = HA02Model()
    
    # Run dummy input to build weights
    batch_size = 8
    dummy_input = tf.random.normal([batch_size, 88, 2])
    output = model(dummy_input, training=False)
    
    # 2. Count parameters
    total_params = model.count_params()
    encoder_params = sum([tf.size(v).numpy() for v in model.encoder.trainable_variables])
    decoder_params = sum([tf.size(v).numpy() for v in model.decoder.trainable_variables])
    
    print(f"\n[Model Summary]")
    print(f" - Encoder (Transformer Stack) Parameters: {encoder_params:,}")
    print(f" - Decoder (Residual Conv) Parameters:     {decoder_params:,}")
    print(f" - Total HA02 Trainable Parameters:        {total_params:,}")
    print(f"   (Reference in paper for 72 pilots: ~105,607 parameters)\n")
    
    print(f"[Forward Pass Verification]")
    print(f" - Input Shape:  {list(dummy_input.shape)}  (Batch, Pilots, Real/Imag)")
    print(f" - Output Shape: {list(output.shape)} (Batch, Symbols=14, Subcarriers=132, Real/Imag)")
    
    # 3. Model Architecture Detail
    print("\n[Architecture Breakdown]")
    print(" 1. Encoder: TransformerEncoderBlock")
    print("    - Linear QKV projection (144 -> 432)")
    print("    - Scaled Dot-Product Multi-Head Attention (N_heads=2)")
    print("    - Add & LayerNorm 1")
    print("    - Feed-Forward Network (144 -> 288 -> 144, GeLU)")
    print("    - Add & LayerNorm 2")
    print(" 2. Decoder: ResidualConvDecoderBlock")
    print("    - Conv2D (1 -> 2 filters, kernel 2x2, padding same)")
    print("    - Residual Block (Conv2D -> ReLU -> Conv2D + BatchNorm)")
    print("    - Upsampling Dense Layer (88 pilots -> 1848 total grid elements)")
    print("    - Conv2D Output Layer (2 -> 1 filter, kernel 2x2, padding same)")
    print("    - Reshape to (14 symbols, 132 subcarriers, 2 channels)")
    print("=" * 70)

