"""
====================================================================================================
CORAL Domain Adaptation with HA02 Attention Model for NTN Channel Estimation (OpenNTN & MATLAB)
====================================================================================================

Overview:
---------
This script trains and evaluates the HA02 Transformer-Convolutional Attention Model with Correlation 
Alignment (CORAL) Unsupervised Domain Adaptation (UDA) for 5G Non-Terrestrial Network (NTN) 
channel estimation.

Architecture (HA02):
--------------------
- Input:  (Batch, 88, 2)       --> Sparse LS estimates at pilot locations (Real & Imag)
- Output: (Batch, 132, 14, 2)  --> Reconstructed full 2D channel grid (Subcarriers x Symbols x 2)
- Modules:
    1. TransformerEncoderBlock: Multi-Head Self-Attention on sparse pilot features + LayerNorm + FFN
    2. ResidualConvDecoderBlock: Conv2D -> ResBlock (Conv-ReLU-Conv+BatchNorm) -> Dense Upsample (88 -> 1848) -> Conv2D

CORAL Layer Extraction:
-----------------------
Users can specify intermediate feature layers to extract and align via `--coral-layers`:
- `layer1`: Transformer Encoder output (Z_enc) -> Shape: [B, 176] (Aligns pilot self-attention features)
- `layer2`: Post-ResConv Decoder feature (Z_conv) -> Shape: [B, 352] (Aligns refined multi-channel features)
- `layer3`: Latent Grid representation (Z_grid) -> Shape: [B, 2] (Aligns global pooled reconstructed grid)

Adaptive Dataset Directory Input Parser (`--source-dir` & `--target-dir`):
-------------------------------------------------------------------------
The input parser supports 3 flexible input formats without requiring you to specify whether a dataset 
is from MATLAB or OpenNTN:

1. Scenario Folder Name / Substring (Auto-Discovered):
   You can pass just the folder name or scenario substring. The loader automatically scans both 
   `generatedChan/MATLAB/` and `generatedChan/OpenNTN/` to find the matching directory:
     --source-dir A100_2p18e9_600km_70deg_30kHz
     --target-dir DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps

2. Relative Path from Project Root:
   You can pass the relative path:
     --source-dir generatedChan/MATLAB/A100_2p18e9_600km_70deg_30kHz
     --target-dir generatedChan/OpenNTN/DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps

3. Full Absolute Path:
   You can pass the full path on disk:
     --source-dir "C:/Users/.../generatedChan/MATLAB/A100_2p18e9_600km_70deg_30kHz"
     --target-dir "C:/Users/.../generatedChan/OpenNTN/DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps"

How the Loader Resolves Files:
- Automatically maps requested `--snr` (e.g. 5) to matching subfolder (`SNR_5dB`, `5dB`, `SNR_5`, `5`, etc.).
- Automatically detects the dataset `.mat` file (`matlabNTN.mat` or `channel_dur_randomizedUE.mat`).
- Supports both MATLAB v7 (scipy.io) and v7.3 HDF5 (h5py) file structures seamlessly.

Examples:
---------
    # Train CORAL UDA on MATLAB A100 dataset vs OpenNTN dataset at SNR = 5 dB (using short scenario names)
    python run_CORAL_LS_Attention.py --source-dir A100_2p18e9_600km_70deg_30kHz --target-dir DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps --snr 5

    # Train CORAL UDA on OpenNTN datasets with feature saving
    python run_CORAL_LS_Attention.py --snr 5 --coral-layers layer1 layer2 --domain-weight 0.5 --save-features

    # Train source-only baseline (no domain adaptation)
    python run_CORAL_LS_Attention.py --snr 5 --only-source

    # Quick test run (small subset, 5 epochs)
    python run_CORAL_LS_Attention.py --test-code --coral-layers layer1 layer2
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
import re
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
SOURCE_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz"
TARGET_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps"
DEFAULT_SAVE_DIR = ""          # Output save directory (defaults to './results' inside current directory)
DEFAULT_SNR = 5                # Channel SNR in dB
MODEL_TYPE = "LS"              # "LS", "LI", or "Prac"
ONLY_SOURCE = False             # Set True to train only on source (no CORAL)
CORAL_LAYERS = ["layer1", "layer2"]  # Default extracted layers for CORAL
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
        
        # Linear projection to Q, K, V
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
# 3. COMPLETE HA02 MODEL WITH CONFIGURABLE LAYER EXTRACTION
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
# 4. CORAL LOSS COMPUTATION
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
    """Compute CORAL loss for a single layer feature pair."""
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


@tf.function
def compute_coral_loss(source_features: list, target_features: list) -> tf.Tensor:
    """Compute total CORAL loss across all specified layer feature pairs."""
    if len(source_features) == 0:
        return tf.constant(0.0, dtype=tf.float32)
    
    layer_losses = []
    for s_feat, t_feat in zip(source_features, target_features):
        l = coral_loss_single_layer(s_feat, t_feat)
        layer_losses.append(l)
        
    return tf.reduce_mean(layer_losses)


# =============================================================================
# 5. DATA PREPROCESSING & SCALING HELPERS
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


def standardizeScaler_ha02(x, y):
    """
    Sample-wise zero-mean unit-variance standardization:
    - x: (B, 88, 2) pilot inputs
    - y: (B, 132, 14, 2) full channel grids
    """
    B = tf.shape(x)[0]
    x_mean = tf.reduce_mean(x, axis=1)  # (B, 2)
    x_mean_sq = tf.reduce_mean(tf.square(x), axis=1)
    x_var = x_mean_sq - tf.square(x_mean)
    x_std = tf.sqrt(tf.clip_by_value(x_var, 1e-30, tf.float32.max))  # (B, 2)

    x_mean_bc_x = tf.reshape(x_mean, [B, 1, 2])
    scale_bc_x = tf.reshape(x_std, [B, 1, 2])
    x_scaled = (x - x_mean_bc_x) / scale_bc_x

    x_mean_bc_y = tf.reshape(x_mean, [B, 1, 1, 2])
    scale_bc_y = tf.reshape(x_std, [B, 1, 1, 2])
    y_scaled = (y - x_mean_bc_y) / scale_bc_y

    return x_scaled, y_scaled, x_mean, x_std


def deStandardize_ha02(y_scaled, x_mean, x_std):
    """Invert sample-wise standardization for HA02 predicted channel grids."""
    B = tf.shape(y_scaled)[0]
    scale_bc = tf.reshape(x_std, [B, 1, 1, 2])
    shift_bc = tf.reshape(x_mean, [B, 1, 1, 2])
    return y_scaled * scale_bc + shift_bc


def preprocess_batch(H_perf_batch: np.ndarray, H_in_batch: np.ndarray, lower_range: int = -1, standardize: bool = False):
    """Convert complex channel batches to scaled real tensors."""
    y = tf.cast(complx2real(H_perf_batch), tf.float32)
    x = tf.cast(complx2real(H_in_batch), tf.float32)
    if standardize:
        x_sc, y_sc, val1, val2 = standardizeScaler_ha02(x, y)
    else:
        x_sc, y_sc, val1, val2 = minmaxScaler_ha02(x, y, lower_range)
    return x_sc, y_sc, val1, val2


# =============================================================================
# 6. METRIC HELPERS
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
# 7. ADAPTIVE DATASET RESOLUTION & LOADING (MATLAB & OpenNTN Compatible)
# =============================================================================
def find_any_mat_file(base_dir: str) -> str:
    """Recursively search base_dir for the first valid channel .mat file."""
    if not os.path.exists(base_dir):
        return None
    for root, _, files in os.walk(base_dir):
        for f in sorted(files):
            if f.endswith('.mat') and not f.startswith(('inferredChannel', 'testChannel', 'training_history', 'extracted_features', 'synthesized_results')):
                return os.path.join(root, f)
    return None


def get_mat_file(data_root: str, snr: int = 5) -> str:
    """
    Robustly locate the .mat data file for the requested SNR, supporting:
    - SNR folder variations: 'SNR_-10dB', '-10dB', 'SNR_-10', '-10', '5dB', etc.
    - Relative paths from workspace root or script directory
    - Scenario name substring matching (e.g. 'A100_2p18e9...' matching in generatedChan/MATLAB or generatedChan/OpenNTN)
    """
    if os.path.isfile(data_root) and data_root.endswith('.mat'):
        return os.path.abspath(data_root)

    candidate_roots = []
    if data_root:
        if os.path.isabs(data_root):
            candidate_roots.append(data_root)
        else:
            candidate_roots.append(os.path.abspath(data_root))
            candidate_roots.append(os.path.join(project_root, data_root))
            candidate_roots.append(os.path.join(current_dir, data_root))
            
            # Scenario substring search in generatedChan/
            for parent in [os.path.join(project_root, 'generatedChan', 'MATLAB'),
                           os.path.join(project_root, 'generatedChan', 'OpenNTN')]:
                if os.path.isdir(parent):
                    base_name = os.path.basename(data_root.rstrip('\\/'))
                    for entry in os.listdir(parent):
                        if base_name.lower() in entry.lower():
                            candidate_roots.append(os.path.join(parent, entry))

    # Add default fallbacks
    candidate_roots.extend([
        os.path.join(project_root, 'generatedChan', 'OpenNTN', 'DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps'),
        os.path.join(project_root, 'generatedChan', 'OpenNTN', 'DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps'),
        os.path.join(project_root, 'generatedChan', 'MATLAB', 'A100_2p18e9_600km_70deg_30kHz')
    ])

    snr_variations = [
        f"SNR_{snr}dB",
        f"{snr}dB",
        f"SNR_{snr}",
        f"{snr}",
        f"SNR_{snr:02d}dB",
        f"snr_{snr}db"
    ]

    for root in candidate_roots:
        if not os.path.isdir(root):
            continue
        # Try each SNR subfolder variation
        for snr_var in snr_variations:
            snr_dir = os.path.join(root, snr_var)
            if os.path.isdir(snr_dir):
                mat_file = find_any_mat_file(snr_dir)
                if mat_file:
                    return os.path.abspath(mat_file)
        # Direct check inside root if no SNR subfolders
        mat_file = find_any_mat_file(root)
        if mat_file:
            return os.path.abspath(mat_file)

    raise FileNotFoundError(
        f"Could not find any .mat data files for SNR={snr} in any searched location.\n"
        f"Searched roots: {candidate_roots}"
    )


def load_dataset_attention(mat_filepath: str, input_type: str = 'ls') -> dict:
    """
    Load and format MATLAB v7/v7.3 channel data for HA02 Attention Model.
    Supports both OpenNTN (channel_dur_randomizedUE.mat) and MATLAB (matlabNTN.mat).
    """
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
                        elif 'r' in data.dtype.names and 'i' in data.dtype.names:
                            data = data['r'] + 1j * data['i']
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

    p_cols = np.squeeze(mat_dict['pilot_cols']).astype(int) - 1
    p_rows = np.squeeze(mat_dict['pilot_rows']).astype(int) - 1
    mat_dict['pilot_cols'] = p_cols
    mat_dict['pilot_rows'] = p_rows

    # Extract 88 pilot inputs
    input_key_map = {'ls': 'H_ls_pilots', 'prac': 'H_prac', 'li': 'H_li', 'ls_ori': 'H_ls_pilots_ori'}
    target_key = input_key_map.get(input_type, 'H_ls_pilots')
    if target_key not in mat_dict or mat_dict[target_key] is None:
        for alt in ['H_ls_pilots', 'H_ls_pilots_ori', 'H_ls', 'H_li', 'H_prac', 'H_perfect']:
            if alt in mat_dict and mat_dict[alt] is not None:
                target_key = alt
                break

    raw_in = mat_dict[target_key]
    if raw_in.ndim == 3:
        raw_in = format_3d(raw_in)
        # Spatial indexing: dimension 1 is subcarriers (p_rows), dimension 2 is symbols (p_cols)
        H_in = raw_in[:, p_rows, p_cols] if raw_in.shape[1] == 132 else raw_in[:, p_cols, p_rows]
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
# 8. INFERENCE & FEATURE EXTRACTION HELPERS
# =============================================================================
def infer_full_dataset(model, H_perf: np.ndarray, H_in: np.ndarray, batch_size: int = 16, lower_range: int = -1, standardize: bool = False):
    N = H_in.shape[0]
    preds = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_batch = H_in[start:end]
        y_batch = H_perf[start:end]
        
        x_sc, _, x_val1, x_val2 = preprocess_batch(y_batch, x_batch, lower_range, standardize=standardize)
        pred_sc = model(x_sc, training=False)
        if standardize:
            pred_real = deStandardize_ha02(pred_sc, x_val1, x_val2).numpy()
        else:
            pred_real = deMinMax_ha02(pred_sc, x_val1, x_val2, lower_range=lower_range).numpy()
        preds.append(pred_real[..., 0] + 1j * pred_real[..., 1])
    return np.concatenate(preds, axis=0)


def extract_features(model, H_in: np.ndarray, batch_size: int, selected_layers: list, lower_range: int = -1, standardize: bool = False):
    N = H_in.shape[0]
    dummy_y = np.zeros((N, 132, 14), dtype=np.complex64)
    layer_feats = {lyr: [] for lyr in selected_layers}
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_batch = H_in[start:end]
        y_batch = dummy_y[:end - start]
        
        x_sc, _, _, _ = preprocess_batch(y_batch, x_batch, lower_range, standardize=standardize)
        _, feats = model(x_sc, training=False, return_features=True, selected_layers=selected_layers)
        
        for lyr, f_t in zip(selected_layers, feats):
            layer_feats[lyr].append(f_t.numpy())

    return {lyr: np.concatenate(layer_feats[lyr], axis=0) if layer_feats[lyr] else np.empty((0,)) for lyr in selected_layers}


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
        'pilot_rows': p_rows + 1,
        'pilot_cols': p_cols + 1,
        'test_indices': indices,
        'snr': snr,
        'model_type': model_type
    }
    if H_li is not None:
        out['H_LI_test'] = H_li
    savemat(filepath, out)
    print(f"[Save] Exported test MAT file -> {filepath}")


# =============================================================================
# 9. PLOTTING & VISUALIZATION HELPERS
# =============================================================================
def plot_loss_curves(history: dict, save_dir: str):
    """Plot training loss curves across epochs and save to PDF."""
    if not history.get('train_loss'):
        return
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['train_loss'], label='Total Loss', color='blue', lw=2)
    if 'train_est_loss' in history and len(history['train_est_loss']) > 0:
        ax.plot(epochs, history['train_est_loss'], label='Estimation Loss (Source)', color='green', lw=1.5)
    if 'train_coral_loss' in history and len(history['train_coral_loss']) > 0:
        ax.plot(epochs, history['train_coral_loss'], label='CORAL Loss', color='red', lw=1.5, ls='--')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('HA02 CORAL Training Loss Progression', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    fig.tight_layout()
    out_pdf = os.path.join(save_dir, 'loss_total.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"[Save] Exported loss curves plot -> {out_pdf}")


def plot_metric_curves(history: dict, metric_key_prefix: str, ylabel: str, title: str, filename: str, save_dir: str):
    """Plot 4-way metric curves (Source Train, Source Val, Target Train, Target Val) across checkpoints."""
    checkpoints = history.get('eval_epochs', [])
    if not checkpoints or len(checkpoints) == 0:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    plotted = False

    candidates = [
        ('Source Train', [f'{metric_key_prefix}_train_src', f'{metric_key_prefix}_train_src_db', f'{metric_key_prefix}_db_train_src'], 'royalblue', '--'),
        ('Source Val / Test', [f'{metric_key_prefix}_val_src', f'{metric_key_prefix}_val_src_db', f'{metric_key_prefix}_db_val_src'], 'navy', '-'),
        ('Target Train', [f'{metric_key_prefix}_train_tgt', f'{metric_key_prefix}_train_tgt_db', f'{metric_key_prefix}_db_train_tgt'], 'darkorange', '--'),
        ('Target Val / Test', [f'{metric_key_prefix}_val_tgt', f'{metric_key_prefix}_val_tgt_db', f'{metric_key_prefix}_db_val_tgt'], 'crimson', '-'),
    ]

    for label, keys, color, ls in candidates:
        for k in keys:
            if k in history and len(history[k]) > 0:
                ax.plot(checkpoints, history[k], label=label, color=color, lw=1.8, ls=ls)
                plotted = True
                break

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel('Epoch Checkpoint', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best')
    fig.tight_layout()
    out_pdf = os.path.join(save_dir, filename)
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"[Save] Exported {title} plot -> {out_pdf}")


def plot_all_metrics_summary(history: dict, save_dir: str):
    """Generate 2x2 multi-panel figure summarizing Loss, NMSE, MSE, and SSIM."""
    checkpoints = history.get('eval_epochs', [])
    if not checkpoints or len(checkpoints) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: Loss
    epochs_loss = range(1, len(history.get('train_loss', [])) + 1)
    axes[0, 0].plot(epochs_loss, history.get('train_loss', []), label='Total Loss', color='blue', lw=1.8)
    if 'train_est_loss' in history and len(history['train_est_loss']) > 0:
        axes[0, 0].plot(epochs_loss, history['train_est_loss'], label='Est Loss', color='green', lw=1.3)
    if 'train_coral_loss' in history and len(history['train_coral_loss']) > 0:
        axes[0, 0].plot(epochs_loss, history['train_coral_loss'], label='CORAL Loss', color='red', lw=1.3, ls='--')
    axes[0, 0].set_title("Training Loss Progression", fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend()

    # Panel 2: NMSE (dB)
    for k, col, ls, lbl in [('nmse_train_src_db', 'royalblue', '--', 'Source Train'),
                            ('nmse_val_src_db', 'navy', '-', 'Source Val'),
                            ('nmse_train_tgt_db', 'darkorange', '--', 'Target Train'),
                            ('nmse_val_tgt_db', 'crimson', '-', 'Target Val')]:
        if k in history and len(history[k]) > 0:
            axes[0, 1].plot(checkpoints, history[k], label=lbl, color=col, ls=ls, lw=1.6)
    axes[0, 1].set_title("NMSE Progression (dB)", fontweight='bold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("NMSE (dB)")
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].legend()

    # Panel 3: MSE
    for k, col, ls, lbl in [('mse_train_src', 'royalblue', '--', 'Source Train'),
                            ('mse_val_src', 'navy', '-', 'Source Val'),
                            ('mse_train_tgt', 'darkorange', '--', 'Target Train'),
                            ('mse_val_tgt', 'crimson', '-', 'Target Val')]:
        if k in history and len(history[k]) > 0:
            axes[1, 0].plot(checkpoints, history[k], label=lbl, color=col, ls=ls, lw=1.6)
    axes[1, 0].set_title("Mean Squared Error (MSE)", fontweight='bold')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 0].legend()

    # Panel 4: SSIM
    for k, col, ls, lbl in [('ssim_train_src', 'royalblue', '--', 'Source Train'),
                            ('ssim_val_src', 'navy', '-', 'Source Val'),
                            ('ssim_train_tgt', 'darkorange', '--', 'Target Train'),
                            ('ssim_val_tgt', 'crimson', '-', 'Target Val')]:
        if k in history and len(history[k]) > 0:
            axes[1, 1].plot(checkpoints, history[k], label=lbl, color=col, ls=ls, lw=1.6)
    axes[1, 1].set_title("Structural Similarity (SSIM)", fontweight='bold')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("SSIM")
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].legend()

    fig.tight_layout()
    out_pdf = os.path.join(save_dir, 'metrics_summary_2x2.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"[Save] Exported 2x2 metrics summary -> {out_pdf}")


def save_single_reconstruction_pdf(H_true, H_in, H_pred, title_str: str, out_filename: str, save_dir: str,
                                   p_rows: np.ndarray = None, p_cols: np.ndarray = None):
    """Generate side-by-side 2D heatmap PDF of Ground Truth, Input/LS Pilot Grid, and Model Reconstruction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # 1. Ground Truth
    im0 = axes[0].imshow(np.abs(H_true), aspect='auto', cmap='jet')
    axes[0].set_title("Ground Truth |H_true|", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("OFDM Symbol")
    axes[0].set_ylabel("Subcarrier")
    plt.colorbar(im0, ax=axes[0])

    # 2. Input / Pilot grid
    if H_in.ndim == 2 and H_in.shape == (132, 14):
        grid_in = np.abs(H_in)
    elif (H_in.ndim == 1 and len(H_in) == 88) or (H_in.ndim == 2 and H_in.size == 88):
        grid_in = np.zeros((132, 14), dtype=np.float32)
        r = p_rows if p_rows is not None else np.arange(88)
        c = p_cols if p_cols is not None else np.zeros(88, dtype=int)
        grid_in[r, c] = np.abs(np.squeeze(H_in))
    else:
        grid_in = np.abs(H_in) if H_in.ndim == 2 else np.abs(H_in).reshape(132, 14)

    im1 = axes[1].imshow(grid_in, aspect='auto', cmap='jet')
    axes[1].set_title("Input Channel |H_in|", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("OFDM Symbol")
    axes[1].set_ylabel("Subcarrier")
    plt.colorbar(im1, ax=axes[1])

    # 3. Model Output
    im2 = axes[2].imshow(np.abs(H_pred), aspect='auto', cmap='jet')
    nmse_val = 10.0 * np.log10(compute_nmse(H_pred, H_true) + 1e-30)
    ssim_val = compute_ssim_batch(H_pred[None, ...], H_true[None, ...])
    axes[2].set_title(f"HA02 Output |H_pred|\n(NMSE: {nmse_val:.2f} dB, SSIM: {ssim_val:.4f})", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("OFDM Symbol")
    axes[2].set_ylabel("Subcarrier")
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(title_str, fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    out_pdf = os.path.join(save_dir, out_filename)
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"[Save] Exported channel reconstruction -> {out_pdf}")


# =============================================================================
# 10. MAIN TRAINING AND EVALUATION ROUTINE
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="HA02 Attention Model with Multi-Layer CORAL UDA for OpenNTN & MATLAB datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--source-dir', type=str, default=SOURCE_DIR, help="Source dataset path, scenario name, or directory")
    parser.add_argument('--target-dir', type=str, default=TARGET_DIR, help="Target dataset path, scenario name, or directory")
    parser.add_argument('--save-dir', type=str, default=DEFAULT_SAVE_DIR, help="Directory to save output files")
    parser.add_argument('--snr', type=int, default=DEFAULT_SNR, help="Channel SNR in dB")
    parser.add_argument('--type', type=str, default=MODEL_TYPE, choices=['LS', 'LI', 'Prac', 'ls', 'li', 'prac'], help="Input type")
    parser.add_argument('--only-source', action='store_true', default=ONLY_SOURCE, help="Train only on source data (no CORAL)")
    parser.add_argument(
        '--coral-layers', 
        nargs='+', 
        default=CORAL_LAYERS,
        help="Layers to extract for CORAL alignment (e.g. '--coral-layers layer1 layer2' or '--coral-layers layer1')"
    )
    parser.add_argument('--domain-weight', type=float, default=0.5, help="CORAL loss weight (lambda)")
    parser.add_argument('--ssim-weight', type=float, default=0.1, help="SSIM loss weight (alpha)")
    parser.add_argument('--train-frac', type=float, default=DEFAULT_TRAIN_FRAC, help="Fraction of data for training")
    parser.add_argument('--val-frac', type=float, default=DEFAULT_VAL_FRAC, help="Fraction of data for validation")
    parser.add_argument('--n-epochs', type=int, default=N_EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--lower-range', type=int, default=-1, choices=[-1, 0], help="Min-max scaling lower range")
    parser.add_argument('--standardize', action='store_true', default=False, help="Use sample-wise zero-mean unit-variance standardization instead of min-max scaling")
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
    src_mat_path = get_mat_file(args.source_dir, args.snr)
    tgt_mat_path = get_mat_file(args.target_dir, args.snr)

    norm_str = "Standardize (Zero-Mean, Unit-Var)" if args.standardize else f"Min-Max [{args.lower_range}, 1]"
    print("=" * 80)
    print(f"HA02 Attention Model | Mode: {'Source-Only' if args.only_source else 'Multi-Layer CORAL UDA'}")
    if not args.only_source:
        print(f"CORAL Extracted Layers: {selected_layers} ({len(selected_layers)} Alignment Points)")
        print(f"CORAL Loss Weight (lambda): {domain_weight}")
    print(f"Source Dataset: {src_mat_path}")
    print(f"Target Dataset: {tgt_mat_path}")
    print(f"SNR: {args.snr} dB | Input Type: {args.type} | Normalization: {norm_str}")
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

    # Instantiate HA02 Model
    model = HA02Model(num_pilot_elems=88, total_grid_elems=1848, num_channels=2, num_heads=2, n_filter=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    history = {
        'train_loss': [],
        'train_est_loss': [],
        'train_coral_loss': [],
        'eval_epochs': [],
        # NMSE (dB)
        'nmse_train_src_db': [],
        'nmse_val_src_db': [],
        'nmse_train_tgt_db': [],
        'nmse_val_tgt_db': [],
        # MSE
        'mse_train_src': [],
        'mse_val_src': [],
        'mse_train_tgt': [],
        'mse_val_tgt': [],
        # SSIM
        'ssim_train_src': [],
        'ssim_val_src': [],
        'ssim_train_tgt': [],
        'ssim_val_tgt': []
    }

    saved_features = {}
    mid_epoch = args.n_epochs // 2
    feature_checkpoint_epochs = {0: 'begin', mid_epoch: 'mid', args.n_epochs - 1: 'last'}

    # =========================================================================
    # COMPILED GPU TRAINING STEPS
    # =========================================================================
    ssim_max_val = 6.0 if args.standardize else (2.0 if args.lower_range == -1 else 1.0)

    @tf.function
    def _train_step_coral(x_src, y_src, x_tgt):
        with tf.GradientTape() as tape:
            # 1. Forward pass source domain
            y_pred_src, feats_src = model(x_src, training=True, return_features=True, selected_layers=selected_layers)
            
            # 2. Forward pass target domain
            _, feats_tgt = model(x_tgt, training=True, return_features=True, selected_layers=selected_layers)
            
            # 3. Source Estimation Loss (MSE + SSIM)
            mse_loss = tf.reduce_mean(tf.square(y_src - y_pred_src))
            ssim_loss = 1.0 - tf.reduce_mean(tf_ssim(y_src, y_pred_src, max_val=ssim_max_val))
            est_loss = mse_loss + args.ssim_weight * ssim_loss
            
            # 4. Multi-Layer CORAL Loss
            coral_loss = compute_coral_loss(feats_src, feats_tgt) if domain_weight > 0 else tf.constant(0.0, dtype=tf.float32)
            
            total_loss = est_loss + domain_weight * coral_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, est_loss, coral_loss

    @tf.function
    def _train_step_source_only(x_src, y_src):
        with tf.GradientTape() as tape:
            y_pred_src = model(x_src, training=True)
            mse_loss = tf.reduce_mean(tf.square(y_src - y_pred_src))
            ssim_loss = 1.0 - tf.reduce_mean(tf_ssim(y_src, y_pred_src, max_val=ssim_max_val))
            total_loss = mse_loss + args.ssim_weight * ssim_loss
            
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, total_loss, tf.constant(0.0, dtype=tf.float32)

    print(f"\n[Train] Starting Multi-Layer CORAL Training for {args.n_epochs} Epochs ...")
    start_time = time.perf_counter()

    eval_n_tr = min(len(idx_train_src), 64)
    eval_sub_src_tr = idx_train_src[:eval_n_tr]
    eval_sub_tgt_tr = idx_train_tgt[:eval_n_tr]

    for epoch in range(args.n_epochs):
        # Feature Checkpointing at begin, mid, last
        if args.save_features and epoch in feature_checkpoint_epochs:
            stage = feature_checkpoint_epochs[epoch]
            src_f = extract_features(model, H_in_src[idx_train_src], args.batch_size, selected_layers, args.lower_range, standardize=args.standardize)
            tgt_f = extract_features(model, H_in_tgt[idx_train_tgt], args.batch_size, selected_layers, args.lower_range, standardize=args.standardize)
            for k, v in src_f.items():
                saved_features[f"features_{stage}_{k}_src"] = v
            for k, v in tgt_f.items():
                saved_features[f"features_{stage}_{k}_tgt"] = v
            print(f"  [Features Saved] Captured intermediate activations at {stage} epoch ({epoch+1}) for layers: {selected_layers}")

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
            
            x_src_b, y_src_b, _, _ = preprocess_batch(train_perf_src[s_idx:e_idx], train_in_src[s_idx:e_idx], args.lower_range, standardize=args.standardize)
            x_tgt_b, y_tgt_b, _, _ = preprocess_batch(train_perf_tgt[s_idx:e_idx], train_in_tgt[s_idx:e_idx], args.lower_range, standardize=args.standardize)

            if args.only_source:
                l_tot, l_est, l_cor = _train_step_source_only(x_src_b, y_src_b)
            else:
                l_tot, l_est, l_cor = _train_step_coral(x_src_b, y_src_b, x_tgt_b)

            ep_loss += float(l_tot)
            ep_est += float(l_est)
            ep_coral += float(l_cor)

        avg_loss = ep_loss / max(n_batches, 1)
        avg_est = ep_est / max(n_batches, 1)
        avg_coral = ep_coral / max(n_batches, 1)

        history['train_loss'].append(avg_loss)
        history['train_est_loss'].append(avg_est)
        history['train_coral_loss'].append(avg_coral)

        # Track metrics (NMSE, MSE, SSIM) on train & val splits
        eval_interval = 1 if (args.n_epochs <= 50 or args.test_code) else 5
        if (epoch + 1) % eval_interval == 0 or epoch == args.n_epochs - 1:
            pred_src_tr = infer_full_dataset(model, H_perf_src[eval_sub_src_tr], H_in_src[eval_sub_src_tr], args.batch_size, args.lower_range, standardize=args.standardize)
            pred_src_val = infer_full_dataset(model, H_perf_src[idx_val_src], H_in_src[idx_val_src], args.batch_size, args.lower_range, standardize=args.standardize)
            pred_tgt_tr = infer_full_dataset(model, H_perf_tgt[eval_sub_tgt_tr], H_in_tgt[eval_sub_tgt_tr], args.batch_size, args.lower_range, standardize=args.standardize)
            pred_tgt_val = infer_full_dataset(model, H_perf_tgt[idx_val_tgt], H_in_tgt[idx_val_tgt], args.batch_size, args.lower_range, standardize=args.standardize)

            nmse_s_tr = compute_nmse_db(pred_src_tr, H_perf_src[eval_sub_src_tr])
            nmse_s_val = compute_nmse_db(pred_src_val, H_perf_src[idx_val_src])
            nmse_t_tr = compute_nmse_db(pred_tgt_tr, H_perf_tgt[eval_sub_tgt_tr])
            nmse_t_val = compute_nmse_db(pred_tgt_val, H_perf_tgt[idx_val_tgt])

            mse_s_tr = compute_mmse(pred_src_tr, H_perf_src[eval_sub_src_tr])
            mse_s_val = compute_mmse(pred_src_val, H_perf_src[idx_val_src])
            mse_t_tr = compute_mmse(pred_tgt_tr, H_perf_tgt[eval_sub_tgt_tr])
            mse_t_val = compute_mmse(pred_tgt_val, H_perf_tgt[idx_val_tgt])

            ssim_s_tr = compute_ssim_batch(pred_src_tr, H_perf_src[eval_sub_src_tr])
            ssim_s_val = compute_ssim_batch(pred_src_val, H_perf_src[idx_val_src])
            ssim_t_tr = compute_ssim_batch(pred_tgt_tr, H_perf_tgt[eval_sub_tgt_tr])
            ssim_t_val = compute_ssim_batch(pred_tgt_val, H_perf_tgt[idx_val_tgt])

            history['eval_epochs'].append(epoch + 1)
            history['nmse_train_src_db'].append(nmse_s_tr)
            history['nmse_val_src_db'].append(nmse_s_val)
            history['nmse_train_tgt_db'].append(nmse_t_tr)
            history['nmse_val_tgt_db'].append(nmse_t_val)

            history['mse_train_src'].append(mse_s_tr)
            history['mse_val_src'].append(mse_s_val)
            history['mse_train_tgt'].append(mse_t_tr)
            history['mse_val_tgt'].append(mse_t_val)

            history['ssim_train_src'].append(ssim_s_tr)
            history['ssim_val_src'].append(ssim_s_val)
            history['ssim_train_tgt'].append(ssim_t_tr)
            history['ssim_val_tgt'].append(ssim_t_val)

            print(f"Epoch {epoch+1:03d}/{args.n_epochs:03d} | Loss: {avg_loss:.4f} (Est: {avg_est:.4f}, CORAL: {avg_coral:.4f}) | "
                  f"Target NMSE: {nmse_t_val:.2f} dB (Src: {nmse_s_val:.2f} dB) | Target SSIM: {ssim_t_val:.4f}")

    total_time = time.perf_counter() - start_time
    print(f"\n[Done] Training completed in {total_time:.2f} seconds ({total_time / args.n_epochs:.3f} s/epoch).")

    # =========================================================================
    # FINAL TEST EVALUATION & EXPORTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("                      FINAL TEST PERFORMANCE SUMMARY                  ")
    print("=" * 80)

    # 1. Source and Target Test Predictions
    test_pred_src = infer_full_dataset(model, H_perf_src[idx_test_src], H_in_src[idx_test_src], args.batch_size, args.lower_range, standardize=args.standardize)
    test_nmse_db_src = compute_nmse_db(test_pred_src, H_perf_src[idx_test_src])
    test_mmse_src = compute_mmse(test_pred_src, H_perf_src[idx_test_src])
    test_ssim_src = compute_ssim_batch(test_pred_src, H_perf_src[idx_test_src])

    test_pred_tgt = infer_full_dataset(model, H_perf_tgt[idx_test_tgt], H_in_tgt[idx_test_tgt], args.batch_size, args.lower_range, standardize=args.standardize)
    test_nmse_db_tgt = compute_nmse_db(test_pred_tgt, H_perf_tgt[idx_test_tgt])
    test_mmse_tgt = compute_mmse(test_pred_tgt, H_perf_tgt[idx_test_tgt])
    test_ssim_tgt = compute_ssim_batch(test_pred_tgt, H_perf_tgt[idx_test_tgt])

    train_pred_src_sample = infer_full_dataset(model, H_perf_src[idx_train_src[:10]], H_in_src[idx_train_src[:10]], args.batch_size, args.lower_range, standardize=args.standardize)
    train_pred_tgt_sample = infer_full_dataset(model, H_perf_tgt[idx_train_tgt[:10]], H_in_tgt[idx_train_tgt][:10], args.batch_size, args.lower_range, standardize=args.standardize)

    print(f"  Source Domain -> NMSE: {test_nmse_db_src:.2f} dB | MMSE: {test_mmse_src:.6e} | SSIM: {test_ssim_src:.4f}")
    print(f"  Target Domain -> NMSE: {test_nmse_db_tgt:.2f} dB | MMSE: {test_mmse_tgt:.6e} | SSIM: {test_ssim_tgt:.4f}")
    print("=" * 80)

    # 2. Save testChannel_source.mat and testChannel_target.mat
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

    # 3. Save Plotted Channel Samples to sample_reconstructions.mat
    samples_dict = {
        'source_train_true': H_perf_src[idx_train_src[0]],
        'source_train_in':   H_in_src[idx_train_src[0]],
        'source_train_pred': train_pred_src_sample[0],

        'source_test_true':  H_perf_src[idx_test_src[0]],
        'source_test_in':    H_in_src[idx_test_src[0]],
        'source_test_pred':  test_pred_src[0],

        'target_train_true': H_perf_tgt[idx_train_tgt[0]],
        'target_train_in':   H_in_tgt[idx_train_tgt[0]],
        'target_train_pred': train_pred_tgt_sample[0],

        'target_test_true':  H_perf_tgt[idx_test_tgt[0]],
        'target_test_in':    H_in_tgt[idx_test_tgt[0]],
        'target_test_pred':  test_pred_tgt[0],

        'pilot_rows': p_rows + 1,
        'pilot_cols': p_cols + 1,
        'snr': args.snr,
        'input_type': args.type,
        'model_type': 'HA02'
    }
    savemat(os.path.join(output_dir, 'sample_reconstructions.mat'), samples_dict)
    print(f"[Save] Exported sample reconstruction grids MAT file -> {os.path.join(output_dir, 'sample_reconstructions.mat')}")

    # 4. Save final_epoch.txt Report
    txt_path = os.path.join(output_dir, 'final_epoch.txt')
    try:
        with open(txt_path, 'w') as f:
            f.write("=== FINAL EPOCH EVALUATION RESULTS ===\n")
            f.write(f"SNR (dB):             {args.snr}\n")
            f.write(f"Input Type:           {args.type}\n")
            f.write(f"Normalization:        {norm_str}\n")
            f.write(f"Domain Adaptation:    {'Source-Only' if args.only_source else 'CORAL UDA'}\n")
            f.write(f"CORAL Layers:         {selected_layers}\n")
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
        'standardize': args.standardize,
        'coral_layers': np.array(selected_layers)
    }
    savemat(eval_path, eval_dict)
    print(f"[Save] Evaluation results -> {eval_path}")

    # 6. Save Training History MAT
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
        print(f"[Save] Exported extracted features -> {feat_save_path}")

    # 8. Save All Visualizations
    try:
        # A. Training Loss Progression (Total, Est, CORAL)
        plot_loss_curves(history, output_dir)

        # B. NMSE (dB) curves (Source Train, Source Val, Target Train, Target Val)
        plot_metric_curves(history, 'nmse', 'NMSE (dB)', 'NMSE (dB) Across Epochs', 'metrics_nmse_db.pdf', output_dir)

        # C. MSE curves (Source Train, Source Val, Target Train, Target Val)
        plot_metric_curves(history, 'mse', 'MSE', 'Mean Squared Error (MSE) Across Epochs', 'metrics_mse.pdf', output_dir)

        # D. SSIM curves (Source Train, Source Val, Target Train, Target Val)
        plot_metric_curves(history, 'ssim', 'SSIM', 'Structural Similarity (SSIM) Across Epochs', 'metrics_ssim.pdf', output_dir)

        # E. Consolidated 2x2 Metrics Summary
        plot_all_metrics_summary(history, output_dir)

        # F. Individual Channel Reconstructions (Ground Truth, Input/LS Pilots, Output)
        save_single_reconstruction_pdf(samples_dict['source_train_true'], samples_dict['source_train_in'], samples_dict['source_train_pred'],
                                       "Source Domain - Training Sample", "recon_source_train.pdf", output_dir, p_rows, p_cols)
        save_single_reconstruction_pdf(samples_dict['source_test_true'], samples_dict['source_test_in'], samples_dict['source_test_pred'],
                                       "Source Domain - Testing Sample", "recon_source_test.pdf", output_dir, p_rows, p_cols)
        save_single_reconstruction_pdf(samples_dict['target_train_true'], samples_dict['target_train_in'], samples_dict['target_train_pred'],
                                       "Target Domain - Training Sample", "recon_target_train.pdf", output_dir, p_rows, p_cols)
        save_single_reconstruction_pdf(samples_dict['target_test_true'], samples_dict['target_test_in'], samples_dict['target_test_pred'],
                                       "Target Domain - Testing Sample", "recon_target_test.pdf", output_dir, p_rows, p_cols)

    except Exception as e:
        print(f"[Plot Warning] Failed to render PDF plots: {e}")

    print(f"\n[Done] Finished CORAL training and evaluation. Results saved in: {output_dir}")


if __name__ == '__main__':
    main()
