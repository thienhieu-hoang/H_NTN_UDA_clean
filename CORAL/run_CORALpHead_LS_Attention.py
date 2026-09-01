"""
====================================================================================================
CORAL with Multi-Layer Projection Heads for HA02 Attention Model (OpenNTN)
====================================================================================================

Overview:
---------
This script trains and evaluates the HA02 Transformer-Convolutional Attention Model with 
Non-Linear Projection Head Correlation Alignment (CORAL) Unsupervised Domain Adaptation (UDA)
for 5G Non-Terrestrial Network (NTN) channel estimation.

Projection Head Alignment Architecture:
---------------------------------------
Instead of calculating CORAL loss directly on raw intermediate representations, each extracted
layer is routed through an adaptive, dedicated Non-Linear Projection Head network:
    
    Z_layer [B, d_in] ---> [Dense(hidden_dim) -> LayerNorm -> GeLU -> Dense(proj_dim)] ---> P_layer [B, proj_dim]
                                                                                                 │
                                                                         CORAL Loss: Cov(P_src, P_tgt)

Adaptive Multi-Head Manager:
----------------------------
- If `--coral-layers layer1` is specified:
    --> Instantiates 1 Projection Head network: ProjHead_Layer1 (176 -> 128 -> 64)
- If `--coral-layers layer1 layer2` is specified:
    --> Instantiates 2 distinct Projection Head networks:
        1. ProjHead_Layer1: (176 -> 128 -> 64) [Aligns Transformer Attention Latent Space]
        2. ProjHead_Layer2: (352 -> 128 -> 64) [Aligns Residual Conv Decoder Latent Space]
- If `--coral-layers layer1 layer2 layer3` is specified:
    --> Instantiates 3 distinct Projection Head networks.

During testing/inference, the projection heads are completely bypassed, incurring ZERO runtime overhead.

Usage Examples:
---------------
    # Multi-layer Projection Head CORAL (Layer 1 + Layer 2 with 64-D projected subspace at 5 dB)
    python run_CORALpHead_LS_Attention.py --snr 5 --coral-layers layer1 layer2 --domain-weight 0.5 --save-features

    # Single-layer Projection Head on Transformer Encoder (Layer 1 only)
    python run_CORALpHead_LS_Attention.py --snr 5 --coral-layers layer1 --domain-weight 0.5

    # Source-only baseline (no domain adaptation)
    python run_CORALpHead_LS_Attention.py --snr 5 --only-source

    # Quick test run (small subset, 5 epochs)
    python run_CORALpHead_LS_Attention.py --test-code --coral-layers layer1 layer2
====================================================================================================
"""

# NumPy compatibility patch for NumPy 2.x
import numpy as np
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'string_'):
    np.string_ = np.bytes_
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

try:
    if hasattr(np, 'sctypeDict'):
        if 'string_' not in np.sctypeDict:
            np.sctypeDict['string_'] = np.bytes_
        if 'unicode_' not in np.sctypeDict:
            np.sctypeDict['unicode_'] = np.str_
    if hasattr(np, 'typeDict'):
        if 'string_' not in np.typeDict:
            np.typeDict['string_'] = np.bytes_
        if 'unicode_' not in np.typeDict:
            np.typeDict['unicode_'] = np.str_
except Exception:
    pass

import os
import sys
import time
import argparse
import scipy
from scipy.io import savemat, loadmat
import h5py
import tensorflow as tf
from tensorflow.image import ssim as tf_ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
SOURCE_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps"
TARGET_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps"
DEFAULT_SAVE_DIR = ""          # Output save directory (defaults to './results' inside current directory)
DEFAULT_SNR = 5                # Channel SNR in dB
MODEL_TYPE = "LS"              # "LS", "LI", or "Prac"
ONLY_SOURCE = False             # Set True to train only on source (no CORAL)
CORAL_LAYERS = ["layer1", "layer2"]  # Default extracted layers for Projection Head CORAL
PROJ_DIM = 64                  # Output dimension of each projection head
SAVE_FEATURES = False           # Set True to save extracted features at begin, mid, last epochs
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
N_EPOCHS = 300
BATCH_SIZE = 16
TEST_CODE = False
# ============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))


# =============================================================================
# 1. TRANSFORMER ENCODER BLOCK (Attention Pre-processor)
# =============================================================================
class TransformerEncoderBlock(tf.keras.layers.Layer):
    """
    Transformer Encoder Block for HA02 Attention Architecture.
    - Multi-Head Self-Attention on pilot tokens (N_heads = 2).
    - Add & Layer Normalization 1.
    - Feed-Forward Network (Dense -> GeLU -> Dense).
    - Add & Layer Normalization 2.
    """
    def __init__(self, num_pilot_elems=88, num_channels=2, num_heads=2, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.num_pilot_elems = num_pilot_elems  # 88
        self.num_channels = num_channels        # 2 (Real, Imag)
        self.num_heads = num_heads              # 2 heads
        
        self.in_dim = num_pilot_elems * num_channels  # 88 * 2 = 176
        self.head_dim = self.in_dim // num_heads      # 176 / 2 = 88
        
        # Linear projection for Q, K, V
        self.fc1 = tf.keras.layers.Dense(3 * self.in_dim, name="qkv_projection")
        self.fc2 = tf.keras.layers.Dense(self.in_dim, name="attn_out_projection")
        
        # Layer Normalizations
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm_1")
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm_2")
        
        # Feed-Forward Network
        self.ffn_dense1 = tf.keras.layers.Dense(self.in_dim * 2, name="ffn_dense1")
        self.ffn_dense2 = tf.keras.layers.Dense(self.in_dim, name="ffn_dense2")

    def call(self, inputs):
        """
        Input: (B, num_pilot_elems, 2) e.g., (B, 88, 2)
        Output: (out_reshaped [B, 88, 2], z_enc [B, 176])
        """
        B = tf.shape(inputs)[0]
        x_flat = tf.reshape(inputs, [B, self.in_dim])  # (B, 176)
        
        # 1. Linear projection for Q, K, V
        qkv = self.fc1(x_flat)  # (B, 528)
        qkv = tf.reshape(qkv, [B, 3, self.num_heads, self.head_dim])
        Q = qkv[:, 0, :, :]  # (B, 2, 88)
        K = qkv[:, 1, :, :]  # (B, 2, 88)
        V = qkv[:, 2, :, :]  # (B, 2, 88)
        
        # 2. Scaled Dot-Product Attention
        scale = tf.cast(tf.sqrt(self.num_pilot_elems / self.num_heads), dtype=tf.float32)
        Q_exp = tf.expand_dims(Q, axis=-1)  # (B, 2, 88, 1)
        K_exp = tf.expand_dims(K, axis=-2)  # (B, 2, 1, 88)
        
        scores = tf.matmul(Q_exp, K_exp) / scale  # (B, 2, 88, 88)
        attn_weights = tf.nn.softmax(scores, axis=-1)
        
        V_exp = tf.expand_dims(V, axis=-1)  # (B, 2, 88, 1)
        attn_out = tf.squeeze(tf.matmul(attn_weights, V_exp), axis=-1)  # (B, 2, 88)
        
        # 3. Concatenate heads and project back
        attn_out_flat = tf.reshape(attn_out, [B, self.in_dim])  # (B, 176)
        attn_proj = self.fc2(attn_out_flat)                    # (B, 176)
        
        # 4. Residual + LayerNorm 1
        x_norm1 = self.ln1(x_flat + attn_proj)
        
        # 5. Feed-Forward Network + LayerNorm 2
        ffn1 = tf.nn.gelu(self.ffn_dense1(x_norm1))
        ffn_out = self.ffn_dense2(ffn1)
        z_enc = self.ln2(x_norm1 + ffn_out)  # Shape: (B, 176) -> Layer 1 representation
        
        # Reshape to pilot format (B, 88, 2)
        out_reshaped = tf.reshape(z_enc, [B, self.num_pilot_elems, self.num_channels])
        return out_reshaped, z_enc


# =============================================================================
# 2. RESIDUAL CONVOLUTIONAL DECODER BLOCK (Decoder + Upsampler)
# =============================================================================
class ResidualConvDecoderBlock(tf.keras.layers.Layer):
    """
    Residual Convolutional Architecture for HA02 Decoder & Grid Upsampling.
    - Conv2D (2x2 kernel, N_filter=2 channels)
    - Residual Module (Conv2D -> ReLU -> Conv2D + BatchNorm)
    - Upsampling Module (Dense layer projecting 88 pilot dimension -> 1848 full grid dimension)
    - Conv2D output layer (maps 2 filters back to 1 channel)
    - Reshape to 2D channel grid (Subcarriers x Symbols x 2)
    """
    def __init__(self, num_pilot_elems=88, total_grid_elems=1848, n_filter=2, **kwargs):
        super(ResidualConvDecoderBlock, self).__init__(**kwargs)
        self.num_pilot_elems = num_pilot_elems    # 88
        self.total_grid_elems = total_grid_elems  # 132 * 14 = 1848
        self.num_subcarriers = total_grid_elems // 14  # 132
        self.n_filter = n_filter                  # 2
        
        self.conv1 = tf.keras.layers.Conv2D(filters=n_filter, kernel_size=(2, 2), padding='same', name="conv1")
        self.res_conv1 = tf.keras.layers.Conv2D(filters=n_filter, kernel_size=(2, 2), padding='same', name="res_conv1")
        self.relu = tf.keras.layers.ReLU()
        self.res_conv2 = tf.keras.layers.Conv2D(filters=n_filter, kernel_size=(2, 2), padding='same', name="res_conv2")
        self.norm = tf.keras.layers.BatchNormalization(name="batch_norm")
        
        self.fc_upsample = tf.keras.layers.Dense(total_grid_elems, name="fc_upsample")
        self.conv_out = tf.keras.layers.Conv2D(filters=1, kernel_size=(2, 2), padding='same', name="conv_out")

    def call(self, inputs, training=False):
        """
        Input: (B, num_pilot_elems, 2) from Transformer Encoder
        Output: (out_grid [B, 132, 14, 2], z_conv [B, 352], z_grid [B, 2])
        """
        B = tf.shape(inputs)[0]
        x_img = tf.expand_dims(inputs, axis=-1)  # (B, 88, 2, 1)
        
        # Conv 1
        h1 = self.conv1(x_img)  # (B, 88, 2, 2)
        
        # Residual Block
        res = self.res_conv1(h1)
        res = self.relu(res)
        res = self.res_conv2(res)
        h2 = self.norm(h1 + res, training=training)  # (B, 88, 2, 2)
        
        # Layer 2 representation: flattened post-ResConv feature (B, 352)
        z_conv = tf.reshape(h2, [B, -1])  # (B, 352)
        
        # Upsampling via Dense layer along spatial pilot axis
        h2_trans = tf.transpose(h2, [0, 3, 2, 1])  # (B, 2, 2, 88)
        h2_upsampled = self.fc_upsample(h2_trans)   # (B, 2, 2, 1848)
        h2_upsampled = tf.transpose(h2_upsampled, [0, 3, 2, 1])  # (B, 1848, 2, 2)
        
        # Layer 3 representation: Global channel pooled across the 1848 grid elements
        z_grid = tf.reduce_mean(h2_upsampled, axis=[1, 2])  # (B, 2)
        
        # Final Conv layer
        out = self.conv_out(h2_upsampled)  # (B, 1848, 2, 1)
        out = tf.squeeze(out, axis=-1)     # (B, 1848, 2)
        
        # Reshape to (B, num_subcarriers=132, symbols=14, Real/Imag=2)
        out_grid = tf.reshape(out, [B, self.num_subcarriers, 14, 2])
        return out_grid, z_conv, z_grid


# =============================================================================
# 3. NON-LINEAR PROJECTION HEAD (Adaptive Alignment Subspace)
# =============================================================================
class ProjectionHead(tf.keras.layers.Layer):
    """
    Lightweight Non-Linear Projection Head for CORAL Domain Alignment.
    Maps intermediate layer representations (e.g. Layer 1: 176-D or Layer 2: 352-D)
    into a compact, domain-invariant metric space (e.g. 64-D).
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, proj_dim: int = 64, name: str = None, **kwargs):
        super(ProjectionHead, self).__init__(name=name, **kwargs)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        
        self.dense1 = tf.keras.layers.Dense(hidden_dim, name=f"{name}_dense1" if name else "dense1")
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"{name}_ln" if name else "ln")
        self.dense2 = tf.keras.layers.Dense(proj_dim, name=f"{name}_dense2" if name else "dense2")

    def call(self, x, training=False):
        x_flat = tf.reshape(x, [tf.shape(x)[0], -1])  # [B, in_dim]
        h = tf.nn.gelu(self.norm(self.dense1(x_flat))) # [B, hidden_dim]
        p = self.dense2(h)                            # [B, proj_dim]
        return p


# Default input and projection dimension map for HA02 layers
PROJECTION_HEAD_CONFIG = {
    'layer1': {'in_dim': 176, 'hidden_dim': 128, 'proj_dim': 64},
    'layer2': {'in_dim': 352, 'hidden_dim': 128, 'proj_dim': 64},
    'layer3': {'in_dim': 2,   'hidden_dim': 32,  'proj_dim': 32}
}


def build_projection_heads(selected_layers: list, custom_proj_dim: int = 64) -> dict:
    """
    Dynamically instantiate dedicated projection heads for each selected layer.
    """
    heads = {}
    for lyr in selected_layers:
        lyr_clean = str(lyr).strip().lower()
        cfg = PROJECTION_HEAD_CONFIG.get(lyr_clean, {'in_dim': 176, 'hidden_dim': 128, 'proj_dim': custom_proj_dim})
        p_dim = custom_proj_dim if custom_proj_dim is not None else cfg['proj_dim']
        heads[lyr_clean] = ProjectionHead(
            in_dim=cfg['in_dim'],
            hidden_dim=cfg['hidden_dim'],
            proj_dim=p_dim,
            name=f"phead_{lyr_clean}"
        )
    return heads


# =============================================================================
# 4. COMPLETE HA02 MODEL WITH CONFIGURABLE LAYER EXTRACTION
# =============================================================================
class HA02Model(tf.keras.Model):
    """
    HA02 Attention Model with configurable single/multi-layer feature extraction for CORAL.
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

    def call(self, inputs, training=False, return_features=False, selected_layers=None):
        """
        Args:
            inputs: Tensor of shape (B, 88, 2) - sparse LS pilot estimates
            training: Boolean indicating training phase
            return_features: If True, returns (out_grid, list_of_features)
            selected_layers: List of layer names to extract, e.g. ['layer1', 'layer2'] or ['layer1']
        """
        encoder_out, z_enc = self.encoder(inputs)
        out_grid, z_conv, z_grid = self.decoder(encoder_out, training=training)
        
        if return_features:
            feature_dict = {
                'layer1': z_enc,      # [B, 176]
                'layer2': z_conv,     # [B, 352]
                'layer3': z_grid      # [B, 2]
            }
            if selected_layers is None:
                selected_layers = ['layer1', 'layer2']
            
            extracted = []
            for lyr in selected_layers:
                clean_lyr = str(lyr).strip().lower()
                if clean_lyr in feature_dict:
                    extracted.append(feature_dict[clean_lyr])
                else:
                    raise ValueError(f"Unknown layer requested for CORAL: '{lyr}'. Available: {list(feature_dict.keys())}")
            
            return out_grid, extracted
        
        return out_grid


# =============================================================================
# 5. CORAL LOSS COMPUTATION
# =============================================================================
@tf.function
def compute_covariance(features: tf.Tensor) -> tf.Tensor:
    """Compute sample covariance matrix of feature batch Z (B, d)."""
    n = tf.cast(tf.shape(features)[0], tf.float32)
    features_centered = features - tf.reduce_mean(features, axis=0, keepdims=True)
    cov_matrix = tf.matmul(features_centered, features_centered, transpose_a=True) / tf.maximum(n - 1.0, 1.0)
    return cov_matrix


@tf.function
def coral_loss_single_layer(source_feat: tf.Tensor, target_feat: tf.Tensor) -> tf.Tensor:
    """Compute CORAL loss for a single projected feature pair."""
    if len(source_feat.shape) > 2:
        source_feat = tf.reshape(source_feat, [tf.shape(source_feat)[0], -1])
    if len(target_feat.shape) > 2:
        target_feat = tf.reshape(target_feat, [tf.shape(target_feat)[0], -1])
    
    d = tf.cast(tf.shape(source_feat)[1], tf.float32)
    source_cov = compute_covariance(source_feat)
    target_cov = compute_covariance(target_feat)
    
    cov_diff_sq = tf.reduce_sum(tf.square(source_cov - target_cov))
    loss = cov_diff_sq / (4.0 * d * d + 1e-12)
    return loss


# =============================================================================
# 6. DATA PREPROCESSING & SCALING HELPERS
# =============================================================================
def complx2real(x: np.ndarray) -> np.ndarray:
    """Stack real and imaginary components along the last dimension."""
    return np.stack([x.real, x.imag], axis=-1)


def minmaxScaler_ha02(x, y, lower_range=-1):
    """Sample-wise min-max scaling for HA02 pilot inputs (x) and channel grids (y)."""
    B = tf.shape(x)[0]
    x_min = tf.reduce_min(x, axis=1)  # (B, 2)
    x_max = tf.reduce_max(x, axis=1)  # (B, 2)
    
    scale = tf.clip_by_value(x_max - x_min, 1e-30, tf.float32.max)
    
    x_min_bc_x = tf.reshape(x_min, [B, 1, 2])
    scale_bc_x = tf.reshape(scale, [B, 1, 2])
    x_scaled = (x - x_min_bc_x) / scale_bc_x
    
    x_min_bc_y = tf.reshape(x_min, [B, 1, 1, 2])
    scale_bc_y = tf.reshape(scale, [B, 1, 1, 2])
    y_scaled = (y - x_min_bc_y) / scale_bc_y
    
    if lower_range == -1:
        x_scaled = x_scaled * 2.0 - 1.0
        y_scaled = y_scaled * 2.0 - 1.0
        
    return x_scaled, y_scaled, x_min, x_max


def deMinMax_ha02(y_scaled, x_min, x_max, lower_range=-1):
    """Invert sample-wise min-max scaling for HA02 predicted channel grids."""
    B = tf.shape(y_scaled)[0]
    if lower_range == -1:
        y_norm = (y_scaled + 1.0) / 2.0
    else:
        y_norm = y_scaled
        
    scale = (x_max - x_min)
    shift = x_min
    
    scale_bc = tf.reshape(scale, [B, 1, 1, 2])
    shift_bc = tf.reshape(shift, [B, 1, 1, 2])
    
    y_denormed = y_norm * scale_bc + shift_bc
    return y_denormed


def preprocess_batch(H_perf_batch: np.ndarray, H_in_batch: np.ndarray, lower_range: int = -1):
    """Convert complex channel batches to scaled real tensors."""
    y = tf.cast(complx2real(H_perf_batch), tf.float32)
    x = tf.cast(complx2real(H_in_batch), tf.float32)
    x_sc, y_sc, val1, val2 = minmaxScaler_ha02(x, y, lower_range)
    return x_sc, y_sc, val1, val2


# =============================================================================
# 7. METRIC HELPERS
# =============================================================================
def compute_nmse(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    diff_sq = np.sum(np.abs(H_pred - H_true)**2)
    ref_sq = np.sum(np.abs(H_true)**2)
    return float(diff_sq / max(ref_sq, 1e-30))


def compute_nmse_db(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    val = compute_nmse(H_pred, H_true)
    return float(10.0 * np.log10(val + 1e-30))


def compute_mmse(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    return float(np.mean(np.abs(H_pred - H_true)**2))


def compute_ssim_batch(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    real_pred = np.stack([H_pred.real, H_pred.imag], axis=-1)
    real_true = np.stack([H_true.real, H_true.imag], axis=-1)
    ssim_list = []
    for i in range(H_pred.shape[0]):
        val_min = np.min(real_true[i])
        val_max = np.max(real_true[i])
        dr = max(val_max - val_min, 1e-30)
        s = tf_ssim(
            tf.convert_to_tensor(real_true[i:i+1], dtype=tf.float32),
            tf.convert_to_tensor(real_pred[i:i+1], dtype=tf.float32),
            max_val=dr
        )
        ssim_list.append(float(s.numpy()[0]))
    return float(np.mean(ssim_list))


# =============================================================================
# 8. DATASET LOADING & RESOLUTION
# =============================================================================
def get_mat_file(dir_path: str, snr: int = 5) -> str:
    """Resolve dataset path dynamically."""
    if not dir_path or not os.path.exists(dir_path):
        if SOURCE_DIR and os.path.exists(SOURCE_DIR):
            print(f"[Warning] Dataset path '{dir_path}' not found, falling back to '{SOURCE_DIR}'")
            dir_path = SOURCE_DIR
        else:
            return dir_path

    if os.path.isfile(dir_path) and dir_path.endswith('.mat'):
        return dir_path

    candidates = [
        f"{snr}dB", f"SNR_{snr}dB", f"SNR_{snr}", str(snr),
        f"{snr}db", f"snr_{snr}db", f"snr_{snr}"
    ]
    for cand_name in candidates:
        cand_dir = os.path.join(dir_path, cand_name)
        if os.path.exists(cand_dir) and os.path.isdir(cand_dir):
            mat_files = [f for f in os.listdir(cand_dir) if f.endswith('.mat') and not f.startswith(('inferredChannel', 'testChannel', 'training_history', 'extracted_features'))]
            if mat_files:
                return os.path.join(cand_dir, mat_files[0])

    mat_files = [f for f in os.listdir(dir_path) if f.endswith('.mat') and not f.startswith(('inferredChannel', 'testChannel', 'training_history', 'extracted_features'))]
    if mat_files:
        return os.path.join(dir_path, mat_files[0])

    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith('.mat') and not f.startswith(('inferredChannel', 'testChannel', 'training_history', 'extracted_features')):
                return os.path.join(root, f)

    return os.path.join(dir_path, 'matlabNTN.mat')


def load_dataset_attention(mat_filepath: str, input_type: str = 'ls') -> dict:
    """Load and format MATLAB v7/v7.3 channel data for HA02 Attention Model."""
    input_type = input_type.lower()
    mat_dict = {}

    def format_3d(arr):
        if arr is None or not isinstance(arr, np.ndarray) or arr.ndim != 3:
            return arr
        if arr.shape[0] in (14, 132) and arr.shape[-1] not in (14, 132):
            arr = arr.T
        if arr.shape[1] == 14 and arr.shape[2] == 132:
            arr = np.transpose(arr, (0, 2, 1))
        return arr

    if h5py.is_hdf5(mat_filepath):
        with h5py.File(mat_filepath, 'r') as f:
            for k in f.keys():
                if not k.startswith('#'):
                    data = f[k][()]
                    if isinstance(data, np.ndarray) and data.dtype.names is not None:
                        if 'real' in data.dtype.names and 'imag' in data.dtype.names:
                            data = data['real'] + 1j * data['imag']
                    mat_dict[k] = data
    else:
        mat = loadmat(mat_filepath)
        for k, v in mat.items():
            if not k.startswith('__'):
                mat_dict[k] = v

    H_perfect = format_3d(mat_dict.get('H_perfect'))
    mat_dict['H_perfect'] = H_perfect

    H_perfect_ori = None
    for k in ['H_perfect_ori', 'H_perfect_original', 'H_true_ori', 'H_ori']:
        if k in mat_dict and mat_dict[k] is not None:
            H_perfect_ori = format_3d(mat_dict[k])
            break
    mat_dict['H_perfect_ori'] = H_perfect_ori if H_perfect_ori is not None else H_perfect

    p_cols = np.squeeze(mat_dict.get('pilot_cols', np.arange(88)))
    p_rows = np.squeeze(mat_dict.get('pilot_rows', np.arange(88)))
    if np.min(p_cols) >= 1:
        p_cols = p_cols - 1
    if np.min(p_rows) >= 1:
        p_rows = p_rows - 1
    mat_dict['pilot_cols'] = p_cols
    mat_dict['pilot_rows'] = p_rows

    # Extract 88 pilot inputs
    input_key_map = {'ls': 'H_ls_pilots', 'prac': 'H_prac', 'li': 'H_li'}
    target_key = input_key_map.get(input_type, 'H_ls_pilots')
    if target_key not in mat_dict or mat_dict[target_key] is None:
        for alt in ['H_ls_pilots', 'H_ls', 'H_li', 'H_prac', 'H_perfect']:
            if alt in mat_dict and mat_dict[alt] is not None:
                target_key = alt
                break

    raw_in = mat_dict[target_key]
    if raw_in.ndim == 3:
        raw_in = format_3d(raw_in)
        H_in = raw_in[:, p_cols, p_rows] if raw_in.shape[1] == 132 else raw_in[:, p_rows, p_cols]
    elif raw_in.ndim == 2:
        if raw_in.shape[0] == 88 and raw_in.shape[1] == H_perfect.shape[0]:
            H_in = raw_in.T
        else:
            H_in = raw_in
    else:
        H_in = raw_in

    mat_dict['H_input'] = H_in
    if 'H_li' in mat_dict:
        mat_dict['H_li'] = format_3d(mat_dict['H_li'])

    return mat_dict


def split_indices(N: int, train_frac: float = 0.70, val_frac: float = 0.15, seed: int = 1234):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


# =============================================================================
# 9. INFERENCE & FEATURE EXTRACTION HELPERS
# =============================================================================
def infer_full_dataset(model, H_perf: np.ndarray, H_in: np.ndarray, batch_size: int = 16, lower_range: int = -1):
    N = H_in.shape[0]
    preds = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_batch = H_in[start:end]
        y_batch = H_perf[start:end]
        
        x_sc, _, val1, val2 = preprocess_batch(y_batch, x_batch, lower_range)
        out_sc = model(x_sc, training=False)
        out_real = deMinMax_ha02(out_sc, val1, val2, lower_range).numpy()
        out_comp = out_real[..., 0] + 1j * out_real[..., 1]
        preds.append(out_comp)
    return np.concatenate(preds, axis=0)


def extract_features_with_heads(model, proj_heads: dict, H_in: np.ndarray, batch_size: int = 16,
                                selected_layers: list = None, lower_range: int = -1):
    """
    Extract intermediate representations (both raw layer features and projected features).
    Returns dict mapping:
      'layer1' -> raw features [N, 176]
      'phead_layer1' -> projected features [N, proj_dim]
    """
    if selected_layers is None or len(selected_layers) == 0:
        return {}
    
    N = H_in.shape[0]
    raw_feats = {lyr: [] for lyr in selected_layers}
    proj_feats = {lyr: [] for lyr in selected_layers}
    
    dummy_y = np.zeros((batch_size, 132, 14), dtype=np.complex128)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_batch = H_in[start:end]
        y_batch = dummy_y[:end - start]
        
        x_sc, _, _, _ = preprocess_batch(y_batch, x_batch, lower_range)
        _, feats = model(x_sc, training=False, return_features=True, selected_layers=selected_layers)
        
        for lyr, f_t in zip(selected_layers, feats):
            raw_feats[lyr].append(f_t.numpy())
            if lyr in proj_heads:
                p_t = proj_heads[lyr](f_t, training=False)
                proj_feats[lyr].append(p_t.numpy())

    out_dict = {}
    for lyr in selected_layers:
        out_dict[lyr] = np.concatenate(raw_feats[lyr], axis=0) if raw_feats[lyr] else np.empty((0,))
        if lyr in proj_heads and proj_feats[lyr]:
            out_dict[f"phead_{lyr}"] = np.concatenate(proj_feats[lyr], axis=0)
            
    return out_dict


def save_test_channel_mat(filepath: str, H_perf: np.ndarray, H_perf_ori: np.ndarray,
                          H_in: np.ndarray, H_pred: np.ndarray, p_rows: np.ndarray,
                          p_cols: np.ndarray, H_li: np.ndarray, indices: np.ndarray,
                          snr: int, model_type: str):
    """Save test channel representations to MATLAB MAT file matching the standard schema."""
    out = {
        'H_perfect_test': H_perf,
        'H_original_test': H_perf_ori if H_perf_ori is not None else H_perf,
        'H_LS_test': H_in,
        'H_output_test': H_pred,
        'pilot_rows': p_rows + 1 if np.min(p_rows) == 0 else p_rows,
        'pilot_cols': p_cols + 1 if np.min(p_cols) == 0 else p_cols,
        'test_indices': indices,
        'snr': snr,
        'model_type': model_type
    }
    if H_li is not None:
        out['H_LI_test'] = H_li
    savemat(filepath, out)
    print(f"[Save] Exported test MAT file -> {filepath}")


# =============================================================================
# 10. PLOTTING HELPERS
# =============================================================================
def plot_loss_curves(history: dict, save_dir: str):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['train_loss'], label='Total Loss', color='blue', lw=2)
    if 'train_est_loss' in history:
        ax.plot(epochs, history['train_est_loss'], label='Estimation Loss (Source)', color='green', lw=1.5)
    if 'train_coral_loss' in history:
        ax.plot(epochs, history['train_coral_loss'], label='Projected CORAL Loss', color='red', lw=1.5, ls='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('HA02 Projection-Head CORAL Loss Progression')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    out_pdf = os.path.join(save_dir, 'loss_total.pdf')
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_val_curves(history: dict, save_dir: str):
    if not history.get('val_nmse_tgt'):
        return
    epochs = range(1, len(history['val_nmse_tgt']) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'val_nmse_src' in history:
        ax.plot(epochs, history['val_nmse_src'], label='Source Val NMSE', color='blue', lw=1.5)
    ax.plot(epochs, history['val_nmse_tgt'], label='Target Val NMSE', color='orange', lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Validation NMSE (dB) Across Epochs')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    out_pdf = os.path.join(save_dir, 'val_nmse_db.pdf')
    fig.savefig(out_pdf)
    plt.close(fig)


def save_channel_plots_pdf(H_true_sample, H_in_sample, H_pred_sample, save_dir, prefix='target_test'):
    """Side-by-side visualization PDF."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(np.abs(H_true_sample), aspect='auto', cmap='jet')
    axes[0].set_title("Ground Truth Channel |H|")
    axes[0].set_xlabel("OFDM Symbol")
    axes[0].set_ylabel("Subcarrier")
    plt.colorbar(im0, ax=axes[0])

    if H_in_sample.ndim == 2 and H_in_sample.shape == (132, 14):
        im1 = axes[1].imshow(np.abs(H_in_sample), aspect='auto', cmap='jet')
    else:
        im1 = axes[1].plot(np.abs(H_in_sample))
    axes[1].set_title("Input LS Pilot |H_in|")

    im2 = axes[2].imshow(np.abs(H_pred_sample), aspect='auto', cmap='jet')
    axes[2].set_title("HA02 Estimated Channel |H_pred|")
    axes[2].set_xlabel("OFDM Symbol")
    axes[2].set_ylabel("Subcarrier")
    plt.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    out_pdf = os.path.join(save_dir, f'{prefix}_sample_reconstruction.pdf')
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[Save] Exported reconstruction plot -> {out_pdf}")


# =============================================================================
# 11. MAIN TRAINING AND EVALUATION ROUTINE
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="HA02 Attention Model with Multi-Layer Non-Linear Projection Head CORAL UDA for OpenNTN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--source-dir', type=str, default=SOURCE_DIR, help="Source dataset path or directory")
    parser.add_argument('--target-dir', type=str, default=TARGET_DIR, help="Target dataset path or directory")
    parser.add_argument('--save-dir', type=str, default=DEFAULT_SAVE_DIR, help="Directory to save output files")
    parser.add_argument('--snr', type=int, default=DEFAULT_SNR, help="Channel SNR in dB")
    parser.add_argument('--type', type=str, default=MODEL_TYPE, choices=['LS', 'LI', 'Prac', 'ls', 'li', 'prac'], help="Input type")
    parser.add_argument('--only-source', action='store_true', default=ONLY_SOURCE, help="Train only on source data (no CORAL)")
    parser.add_argument(
        '--coral-layers', 
        nargs='+', 
        default=CORAL_LAYERS,
        help="Layers to extract for Projection Head CORAL alignment (e.g. '--coral-layers layer1 layer2' or '--coral-layers layer1')"
    )
    parser.add_argument('--proj-dim', type=int, default=PROJ_DIM, help="Projection Head output dimension (default: 64)")
    parser.add_argument('--domain-weight', type=float, default=0.5, help="CORAL loss weight (lambda)")
    parser.add_argument('--ssim-weight', type=float, default=0.1, help="SSIM loss weight (alpha)")
    parser.add_argument('--train-frac', type=float, default=DEFAULT_TRAIN_FRAC, help="Fraction of data for training")
    parser.add_argument('--val-frac', type=float, default=DEFAULT_VAL_FRAC, help="Fraction of data for validation")
    parser.add_argument('--n-epochs', type=int, default=N_EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--lower-range', type=int, default=-1, choices=[-1, 0], help="Min-max scaling lower range")
    parser.add_argument('--save-features', action='store_true', default=SAVE_FEATURES, help="Extract & save intermediate features")
    parser.add_argument('--test-code', action='store_true', default=TEST_CODE, help="Fast sanity check")
    parser.add_argument('--no-gpu', action='store_true', help="Disable GPU execution")

    args = parser.parse_args()

    if args.no_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("[Config] GPU disabled by user.")

    # Parse and validate extracted layers
    selected_layers = []
    for item in args.coral_layers:
        for sub in str(item).replace(',', ' ').split():
            sub_clean = sub.strip().lower()
            if sub_clean and sub_clean not in selected_layers:
                selected_layers.append(sub_clean)

    valid_layers = ['layer1', 'layer2', 'layer3']
    for lyr in selected_layers:
        if lyr not in valid_layers:
            raise ValueError(f"Invalid layer '{lyr}' in --coral-layers. Valid options are: {valid_layers}")

    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError(f"train_frac ({args.train_frac}) + val_frac ({args.val_frac}) must be < 1.0.")
    test_frac = 1.0 - args.train_frac - args.val_frac

    args.type = args.type.upper()
    domain_weight = 0.0 if args.only_source else args.domain_weight

    # Resolve dataset paths dynamically
    src_mat_path = os.path.abspath(get_mat_file(args.source_dir, args.snr))
    tgt_mat_path = os.path.abspath(get_mat_file(args.target_dir, args.snr))

    print("=" * 80)
    print(f"HA02 Attention Model | Mode: {'Source-Only' if args.only_source else 'Projection Head CORAL UDA'}")
    if not args.only_source:
        print(f"CORAL Extracted Layers: {selected_layers} ({len(selected_layers)} Dedicated Projection Head Networks)")
        print(f"Projection Head Output Dim: {args.proj_dim} (Covariance Size: [{args.proj_dim} x {args.proj_dim}])")
        print(f"CORAL Loss Weight (lambda): {domain_weight}")
    print(f"Source Dataset: {src_mat_path}")
    print(f"Target Dataset: {tgt_mat_path}")
    print(f"SNR: {args.snr} dB | Input Type: {args.type} | Normalization: [{args.lower_range}, 1]")
    print(f"Split: {args.train_frac:.0%} Train / {args.val_frac:.0%} Val / {test_frac:.0%} Test")
    print("=" * 80)

    # Save Directory Setup
    if args.save_dir and args.save_dir.strip():
        output_dir = os.path.abspath(args.save_dir)
    else:
        output_dir = os.path.join(current_dir, 'results')

    os.makedirs(output_dir, exist_ok=True)
    print(f"Experiment results will be saved to: {output_dir}")

    # Load Data
    src_dict = load_dataset_attention(src_mat_path, args.type)
    tgt_dict = load_dataset_attention(tgt_mat_path, args.type)

    H_perf_src = src_dict['H_perfect']
    H_perf_ori_src = src_dict.get('H_perfect_ori', H_perf_src)
    H_in_src = src_dict['H_input']
    H_li_src = src_dict.get('H_li', None)

    H_perf_tgt = tgt_dict['H_perfect']
    H_perf_ori_tgt = tgt_dict.get('H_perfect_ori', H_perf_tgt)
    H_in_tgt = tgt_dict['H_input']
    H_li_tgt = tgt_dict.get('H_li', None)

    p_cols = src_dict['pilot_cols']
    p_rows = src_dict['pilot_rows']

    N_src = H_perf_src.shape[0]
    N_tgt = H_perf_tgt.shape[0]

    # Split Data
    if args.test_code:
        idx_train_src, idx_val_src, idx_test_src = np.arange(0, 64), np.arange(64, 80), np.arange(80, 96)
        idx_train_tgt, idx_val_tgt, idx_test_tgt = np.arange(0, 64), np.arange(64, 80), np.arange(80, 96)
        args.n_epochs = 5
    else:
        idx_train_src, idx_val_src, idx_test_src = split_indices(N_src, args.train_frac, args.val_frac, seed=1234)
        idx_train_tgt, idx_val_tgt, idx_test_tgt = split_indices(N_tgt, args.train_frac, args.val_frac, seed=1234)

    print(f"Source Split -> Train: {len(idx_train_src)} | Val: {len(idx_val_src)} | Test: {len(idx_test_src)}")
    print(f"Target Split -> Train: {len(idx_train_tgt)} | Val: {len(idx_val_tgt)} | Test: {len(idx_test_tgt)}")

    # Instantiate HA02 Model & Adaptive Projection Heads
    model = HA02Model(num_pilot_elems=88, total_grid_elems=1848, num_channels=2, num_heads=2, n_filter=2)
    proj_heads = build_projection_heads(selected_layers, custom_proj_dim=args.proj_dim)
    
    print(f"\n[Model Initialized] HA02 Attention + {len(proj_heads)} Dedicated Projection Head(s):")
    for lyr, head in proj_heads.items():
        print(f"  --> {lyr.upper()} Projection Head: [{head.in_dim} -> {head.hidden_dim} -> {head.proj_dim}]")

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Pre-build model variables with a dummy forward pass
    dummy_x = tf.zeros((args.batch_size, 88, 2), dtype=tf.float32)
    _, dummy_feats = model(dummy_x, training=False, return_features=True, selected_layers=selected_layers)
    for lyr, f_t in zip(selected_layers, dummy_feats):
        proj_heads[lyr](f_t, training=False)

    history = {
        'train_loss': [],
        'train_est_loss': [],
        'train_coral_loss': [],
        'val_nmse_src': [],
        'val_nmse_tgt': []
    }

    saved_features = {}
    mid_epoch = args.n_epochs // 2
    feature_checkpoint_epochs = {0: 'begin', mid_epoch: 'mid', args.n_epochs - 1: 'last'}

    # =========================================================================
    # COMPILED TRAINING STEP WITH PROJECTION HEADS
    # =========================================================================
    @tf.function
    def _train_step_coral_phead(x_src, y_src, x_tgt):
        with tf.GradientTape() as tape:
            # 1. Forward pass source domain
            y_pred_src, raw_feats_src = model(x_src, training=True, return_features=True, selected_layers=selected_layers)
            
            # 2. Forward pass target domain
            _, raw_feats_tgt = model(x_tgt, training=True, return_features=True, selected_layers=selected_layers)
            
            # 3. Source Estimation Loss (MSE + SSIM)
            mse_loss = tf.reduce_mean(tf.square(y_src - y_pred_src))
            ssim_loss = 1.0 - tf.reduce_mean(tf_ssim(y_src, y_pred_src, max_val=2.0))
            est_loss = mse_loss + args.ssim_weight * ssim_loss
            
            # 4. Multi-Head Projected CORAL Loss
            coral_losses = []
            for lyr, z_s, z_t in zip(selected_layers, raw_feats_src, raw_feats_tgt):
                phead = proj_heads[lyr]
                p_s = phead(z_s, training=True)
                p_t = phead(z_t, training=True)
                l_c = coral_loss_single_layer(p_s, p_t)
                coral_losses.append(l_c)
            
            coral_loss = tf.add_n(coral_losses) / tf.cast(len(coral_losses), tf.float32) if coral_losses else tf.constant(0.0)
            
            total_loss = est_loss + domain_weight * coral_loss

        # Collect trainable variables from BOTH main model AND active projection heads
        trainable_vars = list(model.trainable_variables)
        for lyr in selected_layers:
            trainable_vars.extend(proj_heads[lyr].trainable_variables)

        grads = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        return total_loss, est_loss, coral_loss

    @tf.function
    def _train_step_source_only(x_src, y_src):
        with tf.GradientTape() as tape:
            y_pred_src = model(x_src, training=True)
            mse_loss = tf.reduce_mean(tf.square(y_src - y_pred_src))
            ssim_loss = 1.0 - tf.reduce_mean(tf_ssim(y_src, y_pred_src, max_val=2.0))
            total_loss = mse_loss + args.ssim_weight * ssim_loss
            
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, total_loss, tf.constant(0.0, dtype=tf.float32)

    print(f"\n[Train] Starting Projection-Head CORAL Training for {args.n_epochs} Epochs ...")
    start_time = time.perf_counter()

    for epoch in range(args.n_epochs):
        # Feature Checkpointing at begin, mid, last
        if args.save_features and epoch in feature_checkpoint_epochs:
            stage = feature_checkpoint_epochs[epoch]
            src_f = extract_features_with_heads(model, proj_heads, H_in_src[idx_train_src], args.batch_size, selected_layers, args.lower_range)
            tgt_f = extract_features_with_heads(model, proj_heads, H_in_tgt[idx_train_tgt], args.batch_size, selected_layers, args.lower_range)
            for k, v in src_f.items():
                saved_features[f"features_{stage}_{k}_src"] = v
            for k, v in tgt_f.items():
                saved_features[f"features_{stage}_{k}_tgt"] = v
            print(f"  [Features Saved] Captured intermediate & projected activations at {stage} epoch ({epoch+1})")

        # Shuffle training sets
        p_src = np.random.permutation(len(idx_train_src))
        p_tgt = np.random.permutation(len(idx_train_tgt))
        
        train_in_src = H_in_src[idx_train_src][p_src]
        train_perf_src = H_perf_src[idx_train_src][p_src]
        train_in_tgt = H_in_tgt[idx_train_tgt][p_tgt]
        train_perf_tgt = H_perf_tgt[idx_train_tgt][p_tgt]

        n_batches = min(len(idx_train_src), len(idx_train_tgt)) // args.batch_size
        ep_loss, ep_est, ep_coral = 0.0, 0.0, 0.0

        for b in range(n_batches):
            s_idx = b * args.batch_size
            e_idx = s_idx + args.batch_size
            
            x_src_b, y_src_b, _, _ = preprocess_batch(train_perf_src[s_idx:e_idx], train_in_src[s_idx:e_idx], args.lower_range)
            x_tgt_b, y_tgt_b, _, _ = preprocess_batch(train_perf_tgt[s_idx:e_idx], train_in_tgt[s_idx:e_idx], args.lower_range)

            if args.only_source:
                l_tot, l_est, l_cor = _train_step_source_only(x_src_b, y_src_b)
            else:
                l_tot, l_est, l_cor = _train_step_coral_phead(x_src_b, y_src_b, x_tgt_b)

            ep_loss += float(l_tot)
            ep_est += float(l_est)
            ep_coral += float(l_cor)

        avg_loss = ep_loss / max(n_batches, 1)
        avg_est = ep_est / max(n_batches, 1)
        avg_coral = ep_coral / max(n_batches, 1)

        history['train_loss'].append(avg_loss)
        history['train_est_loss'].append(avg_est)
        history['train_coral_loss'].append(avg_coral)

        # Periodic Validation
        pred_val_src = infer_full_dataset(model, H_perf_src[idx_val_src], H_in_src[idx_val_src], args.batch_size, args.lower_range)
        val_nmse_src_db = compute_nmse_db(pred_val_src, H_perf_src[idx_val_src])
        history['val_nmse_src'].append(val_nmse_src_db)

        pred_val_tgt = infer_full_dataset(model, H_perf_tgt[idx_val_tgt], H_in_tgt[idx_val_tgt], args.batch_size, args.lower_range)
        val_nmse_tgt_db = compute_nmse_db(pred_val_tgt, H_perf_tgt[idx_val_tgt])
        history['val_nmse_tgt'].append(val_nmse_tgt_db)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.n_epochs - 1:
            print(f"Epoch {epoch+1:03d}/{args.n_epochs:03d} | Loss: {avg_loss:.4f} (Est: {avg_est:.4f}, Proj-CORAL: {avg_coral:.4f}) | Val Target NMSE: {val_nmse_tgt_db:.2f} dB")

    total_time = time.perf_counter() - start_time
    print(f"\n[Done] Training completed in {total_time:.2f} seconds ({total_time / args.n_epochs:.3f} s/epoch).")

    # =========================================================================
    # FINAL TEST EVALUATION & EXPORTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("                      FINAL TEST PERFORMANCE SUMMARY                  ")
    print("=" * 80)

    # 1. Source Test Evaluation
    test_pred_src = infer_full_dataset(model, H_perf_src[idx_test_src], H_in_src[idx_test_src], args.batch_size, args.lower_range)
    test_nmse_db_src = compute_nmse_db(test_pred_src, H_perf_src[idx_test_src])
    test_mmse_src = compute_mmse(test_pred_src, H_perf_src[idx_test_src])
    test_ssim_src = compute_ssim_batch(test_pred_src, H_perf_src[idx_test_src])
    print(f"  Source Domain -> NMSE: {test_nmse_db_src:.2f} dB | MMSE: {test_mmse_src:.6e} | SSIM: {test_ssim_src:.4f}")

    # 2. Target Test Evaluation
    test_pred_tgt = infer_full_dataset(model, H_perf_tgt[idx_test_tgt], H_in_tgt[idx_test_tgt], args.batch_size, args.lower_range)
    test_nmse_db_tgt = compute_nmse_db(test_pred_tgt, H_perf_tgt[idx_test_tgt])
    test_mmse_tgt = compute_mmse(test_pred_tgt, H_perf_tgt[idx_test_tgt])
    test_ssim_tgt = compute_ssim_batch(test_pred_tgt, H_perf_tgt[idx_test_tgt])
    print(f"  Target Domain -> NMSE: {test_nmse_db_tgt:.2f} dB | MMSE: {test_mmse_tgt:.6e} | SSIM: {test_ssim_tgt:.4f}")
    print("=" * 80)

    # 3. Save testChannel_source.mat and testChannel_target.mat
    save_test_channel_mat(
        os.path.join(output_dir, 'testChannel_source.mat'),
        H_perf_src[idx_test_src],
        H_perf_ori_src[idx_test_src] if H_perf_ori_src is not None else None,
        H_in_src[idx_test_src],
        test_pred_src,
        p_rows, p_cols,
        H_li_src[idx_test_src] if H_li_src is not None else None,
        idx_test_src, args.snr, args.type
    )

    save_test_channel_mat(
        os.path.join(output_dir, 'testChannel_target.mat'),
        H_perf_tgt[idx_test_tgt],
        H_perf_ori_tgt[idx_test_tgt] if H_perf_ori_tgt is not None else None,
        H_in_tgt[idx_test_tgt],
        test_pred_tgt,
        p_rows, p_cols,
        H_li_tgt[idx_test_tgt] if H_li_tgt is not None else None,
        idx_test_tgt, args.snr, args.type
    )

    # 4. Save final_epoch.txt Report
    txt_path = os.path.join(output_dir, 'final_epoch.txt')
    try:
        with open(txt_path, 'w') as f:
            f.write("=== FINAL EPOCH EVALUATION RESULTS ===\n")
            f.write(f"SNR (dB):             {args.snr}\n")
            f.write(f"Input Type:           {args.type}\n")
            f.write(f"Domain Adaptation:    {'Source-Only' if args.only_source else 'Projection Head CORAL UDA'}\n")
            f.write(f"CORAL Layers:         {selected_layers}\n")
            f.write(f"Projection Dim:       {args.proj_dim}\n")
            f.write(f"Total Execution Time: {total_time:.1f} s\n\n")

            f.write("--- SOURCE DOMAIN TEST METRICS ---\n")
            f.write(f"Model Output MMSE:    {test_mmse_src:e}\n")
            f.write(f"Model Output NMSE:    {compute_nmse(test_pred_src, H_perf_src[idx_test_src]):e} ({test_nmse_db_src:.2f} dB)\n")
            f.write(f"Model Output SSIM:    {test_ssim_src:.4f}\n\n")

            f.write("--- TARGET DOMAIN TEST METRICS ---\n")
            f.write(f"Model Output MMSE:    {test_mmse_tgt:e}\n")
            f.write(f"Model Output NMSE:    {compute_nmse(test_pred_tgt, H_perf_tgt[idx_test_tgt]):e} ({test_nmse_db_tgt:.2f} dB)\n")
            f.write(f"Model Output SSIM:    {test_ssim_tgt:.4f}\n")
        print(f"[Save] Final epoch text report -> {txt_path}")
    except Exception as e:
        print(f"[Save Warning] Failed to write final_epoch.txt: {e}")

    # 5. Save evaluation_results.mat
    eval_path = os.path.join(output_dir, 'evaluation_results.mat')
    eval_dict = {
        'nmse_test_src': compute_nmse(test_pred_src, H_perf_src[idx_test_src]),
        'nmse_test_src_db': test_nmse_db_src,
        'mmse_test_src': test_mmse_src,
        'ssim_test_src': test_ssim_src,
        'nmse_test_tgt': compute_nmse(test_pred_tgt, H_perf_tgt[idx_test_tgt]),
        'nmse_test_tgt_db': test_nmse_db_tgt,
        'mmse_test_tgt': test_mmse_tgt,
        'ssim_test_tgt': test_ssim_tgt,
        'mmse_test': test_mmse_tgt,
        'nmse_test': compute_nmse(test_pred_tgt, H_perf_tgt[idx_test_tgt]),
        'nmse_test_db': test_nmse_db_tgt,
        'ssim_test': test_ssim_tgt,
        'indices_train_source': idx_train_src,
        'indices_val_source': idx_val_src,
        'indices_test_source': idx_test_src,
        'indices_train_target': idx_train_tgt,
        'indices_val_target': idx_val_tgt,
        'indices_test_target': idx_test_tgt,
        'snr': args.snr,
        'input_type': args.type,
        'proj_dim': args.proj_dim,
        'coral_layers': np.array(selected_layers)
    }
    savemat(eval_path, eval_dict)
    print(f"[Save] Evaluation results -> {eval_path}")

    # 6. Save Training History
    history_save_path = os.path.join(output_dir, 'training_history.mat')
    savemat(history_save_path, {k: np.array(v) for k, v in history.items()})
    print(f"[Save] Saved training history -> {history_save_path}")

    # 7. Save Extracted Features MAT if requested
    if args.save_features and saved_features:
        feat_save_path = os.path.join(output_dir, 'extracted_features.mat')
        saved_features['selected_layers'] = np.array(selected_layers)
        saved_features['train_indices_src'] = idx_train_src
        saved_features['train_indices_tgt'] = idx_train_tgt
        savemat(feat_save_path, saved_features, do_compression=True)
        print(f"[Save] Exported extracted raw and projected features -> {feat_save_path}")

    # 8. Save Visualizations
    try:
        plot_loss_curves(history, output_dir)
        plot_val_curves(history, output_dir)
        if len(test_pred_tgt) > 0:
            sample_true = H_perf_tgt[idx_test_tgt[0]]
            sample_in = H_in_tgt[idx_test_tgt[0]]
            save_channel_plots_pdf(sample_true, sample_in, test_pred_tgt[0], output_dir, prefix='target_test')
    except Exception as e:
        print(f"[Plot Warning] Failed to render PDF plots: {e}")

    print(f"\n[Done] Finished Projection-Head CORAL training and evaluation. Results saved in: {output_dir}")


if __name__ == '__main__':
    main()
