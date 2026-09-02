"""
====================================================================================================
CORAL Domain Adaptation with HA02 Attention Model for NTN Channel Estimation (OpenNTN)
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

Examples:
---------
    # Multi-layer CORAL adaptation (Layer 1 + Layer 2)
    python run_CORAL_LS_Attention.py --coral-layers layer1 layer2 --domain-weight 0.5

    # Single-layer CORAL adaptation (Layer 1 only)
    python run_CORAL_LS_Attention.py --coral-layers layer1 --domain-weight 0.5

    # Single-layer CORAL adaptation (Layer 2 only)
    python run_CORAL_LS_Attention.py --coral-layers layer2 --domain-weight 0.5

    # Source-only baseline (no domain adaptation)
    python run_CORAL_LS_Attention.py --only-source

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

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
SOURCE_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps"
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

# Setup project directories and paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
helper_dir = os.path.join(project_root, 'JMMD', 'helper')
domain_helper_dir = os.path.join(project_root, 'Domain_Adversarial', 'helper')
single_dataset_dir = os.path.join(project_root, 'single_dataset')

for path in [domain_helper_dir, project_root, helper_dir, single_dataset_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import helper functions
try:
    import loader
    import plotfig
    import PAD
except ImportError:
    pass


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
def compute_covariance(features: tf.Tensor) -> tf.Tensor:
    """
    Compute sample covariance matrix of feature batch Z (B, d).
    C = (1 / (B - 1)) * (Z - mean(Z))^T * (Z - mean(Z))
    """
    n = tf.cast(tf.shape(features)[0], tf.float32)
    features_centered = features - tf.reduce_mean(features, axis=0, keepdims=True)
    cov_matrix = tf.matmul(features_centered, features_centered, transpose_a=True) / tf.maximum(n - 1.0, 1.0)
    return cov_matrix


def coral_loss_single_layer(source_feat: tf.Tensor, target_feat: tf.Tensor) -> tf.Tensor:
    """
    Compute CORAL loss for a single layer feature pair.
    L_CORAL = (1 / (4 * d^2)) * || C_src - C_tgt ||_F^2
    """
    if len(source_feat.shape) > 2:
        source_feat = tf.reshape(source_feat, [tf.shape(source_feat)[0], -1])
    if len(target_feat.shape) > 2:
        target_feat = tf.reshape(target_feat, [tf.shape(target_feat)[0], -1])
    
    d = tf.cast(tf.shape(source_feat)[1], tf.float32)
    source_cov = compute_covariance(source_feat)
    target_cov = compute_covariance(target_feat)
    
    # Frobenius norm squared of covariance difference
    cov_diff_sq = tf.reduce_sum(tf.square(source_cov - target_cov))
    loss = cov_diff_sq / (4.0 * d * d + 1e-12)
    return loss


def compute_coral_loss(source_features: list, target_features: list) -> tf.Tensor:
    """
    Compute total CORAL loss across all specified layer feature pairs.
    Returns average CORAL loss across the extracted layers.
    """
    if not isinstance(source_features, (list, tuple)):
        source_features = [source_features]
    if not isinstance(target_features, (list, tuple)):
        target_features = [target_features]
    
    if len(source_features) == 0:
        return tf.constant(0.0, dtype=tf.float32)
    
    layer_losses = []
    for s_feat, t_feat in zip(source_features, target_features):
        l_coral = coral_loss_single_layer(s_feat, t_feat)
        layer_losses.append(l_coral)
    
    return tf.reduce_mean(layer_losses)


# =============================================================================
# 5. DATA PREPROCESSING & SCALING HELPERS
# =============================================================================
def complx2real(x: np.ndarray) -> np.ndarray:
    """Stack real and imaginary components along the last dimension."""
    return np.stack([x.real, x.imag], axis=-1)


def minmaxScaler_ha02(x, y, lower_range=-1):
    """
    Sample-wise min-max scaling for HA02 pilot inputs (x) and channel grids (y).
    """
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
    """
    Invert sample-wise min-max scaling for HA02 predicted channel grids.
    """
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
    """Sample-wise standardization for HA02 inputs and grids."""
    B = tf.shape(x)[0]
    mean = tf.reduce_mean(x, axis=1)  # (B, 2)
    std = tf.math.reduce_std(x, axis=1)
    std = tf.clip_by_value(std, 1e-8, tf.float32.max)

    mean_bc_x = tf.reshape(mean, [B, 1, 2])
    std_bc_x = tf.reshape(std, [B, 1, 2])
    x_scaled = (x - mean_bc_x) / std_bc_x

    mean_bc_y = tf.reshape(mean, [B, 1, 1, 2])
    std_bc_y = tf.reshape(std, [B, 1, 1, 2])
    y_scaled = (y - mean_bc_y) / std_bc_y

    return x_scaled, y_scaled, mean, std


def deStandardize_ha02(y_scaled, mean, std):
    """Invert sample-wise standardization."""
    B = tf.shape(y_scaled)[0]
    mean_bc = tf.reshape(mean, [B, 1, 1, 2])
    std_bc = tf.reshape(std, [B, 1, 1, 2])
    return y_scaled * std_bc + mean_bc


def preprocess_batch(H_perf_batch: np.ndarray, H_in_batch: np.ndarray, lower_range: int = -1, standardize: bool = False):
    """Convert complex channel batches to scaled real tensors."""
    y = tf.cast(complx2real(H_perf_batch), tf.float32)
    x = tf.cast(complx2real(H_in_batch), tf.float32)
    if standardize:
        x_sc, y_sc, val1, val2 = standardizeScaler_ha02(x, y)
    else:
        x_sc, y_sc, val1, val2 = minmaxScaler_ha02(x, y, lower_range)
    return x_sc, y_sc, val1, val2


def h5_to_complex(data):
    """Convert HDF5 compound or structured complex data to numpy complex128 array."""
    if isinstance(data, np.ndarray):
        if np.iscomplexobj(data):
            return data
        if data.dtype.names:
            if 'r' in data.dtype.names and 'i' in data.dtype.names:
                return data['r'] + 1j * data['i']
            if 'real' in data.dtype.names and 'imag' in data.dtype.names:
                return data['real'] + 1j * data['imag']
        if data.shape[-1] == 2 and data.ndim >= 2:
            return data[..., 0] + 1j * data[..., 1]
    return np.array(data, dtype=np.complex128)


def get_mat_file(dir_path: str, snr: int = None) -> str:
    """Dynamically and robustly resolve the primary channel .mat dataset filepath in a given directory or SNR subfolder."""
    if not dir_path or not os.path.exists(dir_path):
        if SOURCE_DIR and os.path.exists(SOURCE_DIR):
            print(f"[Warning] Dataset path '{dir_path}' not found, falling back to '{SOURCE_DIR}'")
            dir_path = SOURCE_DIR
        else:
            return dir_path
        
    if os.path.isfile(dir_path) and dir_path.endswith('.mat'):
        return dir_path

    primary_names = ['matlabNTN.mat', 'channel_dur_randomizedUE.mat', 'channel_dur.mat', 'channel_sur.mat', 'channel.mat', 'dataset.mat', 'data.mat']
    excluded_prefixes = ('inferredChannel', 'testChannel', 'training_history', 'extracted_features', 'evaluation_results', 'channel_grids', 'synthesized', 'sample_')

    def find_best_mat_in_dir(folder):
        if not os.path.isdir(folder):
            return None
        # 1. Check primary names
        for name in primary_names:
            p = os.path.join(folder, name)
            if os.path.isfile(p):
                return p
        # 2. Check any valid .mat excluding artifact prefixes
        for f in sorted(os.listdir(folder)):
            if f.endswith('.mat') and not f.startswith(excluded_prefixes):
                return os.path.join(folder, f)
        return None

    # If snr is provided, search SNR subfolders first
    if snr is not None:
        snr_candidates = [
            os.path.join(dir_path, f"SNR_{snr}dB"),
            os.path.join(dir_path, f"{snr}dB"),
            os.path.join(dir_path, f"SNR_{snr}"),
            os.path.join(dir_path, f"{snr}"),
            os.path.join(dir_path, f"SNR_{snr:02d}dB"),
        ]
        for cand in snr_candidates:
            mat = find_best_mat_in_dir(cand)
            if mat:
                return mat

    # Check direct in dir_path
    mat = find_best_mat_in_dir(dir_path)
    if mat:
        return mat

    # Recursive fallback
    for root, _, files in os.walk(dir_path):
        for name in primary_names:
            if name in files:
                return os.path.join(root, name)
        for f in sorted(files):
            if f.endswith('.mat') and not f.startswith(excluded_prefixes):
                return os.path.join(root, f)

    return os.path.join(dir_path, 'matlabNTN.mat')


def load_dataset(mat_filepath: str, input_type: str = 'ls'):
    """
    Load channel data from MATLAB v7.3 (HDF5) or legacy v7 format.
    Returns:
        H_perfect: [N, 132, 14] complex array (Subcarriers x Symbols)
        H_input_pilots: [N, 88] complex array (Sparse pilot measurements)
        pilot_rows: pilot row coordinates
        pilot_cols: pilot col coordinates
        H_perfect_ori: channel before Doppler compensation (or None)
        H_li: benchmark LI channel grid (or None)
        mat_dict: raw dictionary
    """
    input_type = input_type.lower()
    mat_dict = {}

    if h5py.is_hdf5(mat_filepath):
        with h5py.File(mat_filepath, 'r') as f:
            H_perfect = h5_to_complex(f['H_perfect'][()])
            if H_perfect.ndim == 3 and H_perfect.shape[1] == 14 and H_perfect.shape[2] == 132:
                H_perfect = np.transpose(H_perfect, (0, 2, 1))

            H_perfect_ori = None
            for k in ['H_perfect_ori', 'H_perfect_original', 'H_true_ori', 'H_ori']:
                if k in f:
                    H_perfect_ori = h5_to_complex(f[k][()])
                    if H_perfect_ori.ndim == 3 and H_perfect_ori.shape[1] == 14 and H_perfect_ori.shape[2] == 132:
                        H_perfect_ori = np.transpose(H_perfect_ori, (0, 2, 1))
                    break
            if H_perfect_ori is None:
                H_perfect_ori = H_perfect

            for k in f.keys():
                if not k.startswith('#'):
                    mat_dict[k] = h5_to_complex(f[k][()])

            pilot_cols = np.squeeze(mat_dict['pilot_cols'])
            pilot_rows = np.squeeze(mat_dict['pilot_rows'])
            if np.min(pilot_cols) >= 1:
                pilot_cols = pilot_cols - 1
            if np.min(pilot_rows) >= 1:
                pilot_rows = pilot_rows - 1

            input_key_map = {
                'prac': 'H_prac',
                'li': 'H_li',
                'li_ori': 'H_li_ori',
                'ls': 'H_ls_pilots',
                'ls_ori': 'H_ls_pilots_ori'
            }
            target_key = input_key_map.get(input_type, 'H_ls_pilots')
            if target_key not in mat_dict:
                for alt in ['H_ls_pilots', 'H_ls_pilots_ori', 'H_ls', 'H_li', 'H_prac']:
                    if alt in mat_dict:
                        target_key = alt
                        break

            H_in = mat_dict[target_key]
            if H_in.ndim == 3:
                # If a 2D grid was provided instead of 1D pilot vector, extract at pilot positions
                H_in = H_in[:, pilot_cols, pilot_rows]

            H_li = mat_dict.get('H_li', mat_dict.get('H_li_ori', None))
            if H_li is not None and H_li.ndim == 3 and H_li.shape[1] == 14 and H_li.shape[2] == 132:
                H_li = np.transpose(H_li, (0, 2, 1))

    else:
        mat = loadmat(mat_filepath)
        H_perfect = mat['H_perfect'].T if mat['H_perfect'].ndim == 3 else mat['H_perfect']
        if H_perfect.ndim == 3 and H_perfect.shape[1] == 14 and H_perfect.shape[2] == 132:
            H_perfect = np.transpose(H_perfect, (0, 2, 1))

        H_perfect_ori = None
        for k in ['H_perfect_ori', 'H_perfect_original', 'H_true_ori', 'H_ori']:
            if k in mat:
                H_perfect_ori = mat[k].T if mat[k].ndim == 3 else mat[k]
                if H_perfect_ori.ndim == 3 and H_perfect_ori.shape[1] == 14 and H_perfect_ori.shape[2] == 132:
                    H_perfect_ori = np.transpose(H_perfect_ori, (0, 2, 1))
                break
        if H_perfect_ori is None:
            H_perfect_ori = H_perfect

        for k, v in mat.items():
            if not k.startswith('__'):
                mat_dict[k] = v

        pilot_cols = np.squeeze(mat['pilot_cols'])
        pilot_rows = np.squeeze(mat['pilot_rows'])
        if np.min(pilot_cols) >= 1:
            pilot_cols = pilot_cols - 1
        if np.min(pilot_rows) >= 1:
            pilot_rows = pilot_rows - 1

        input_key_map = {
            'prac': 'H_prac',
            'li': 'H_li',
            'li_ori': 'H_li_ori',
            'ls': 'H_ls_pilots',
            'ls_ori': 'H_ls_pilots_ori'
        }
        target_key = input_key_map.get(input_type, 'H_ls_pilots')
        if target_key not in mat or mat[target_key].size == 0:
            for alt in ['H_ls_pilots', 'H_ls_pilots_ori', 'H_ls', 'H_li', 'H_prac']:
                if alt in mat and isinstance(mat[alt], np.ndarray) and mat[alt].size > 0:
                    target_key = alt
                    break

        H_in = mat[target_key].T if mat[target_key].ndim >= 2 and mat[target_key].shape[0] != H_perfect.shape[0] else mat[target_key]
        if H_in.ndim == 3:
            H_in = H_in[:, pilot_cols, pilot_rows]

        H_li = mat.get('H_li', mat.get('H_li_ori', None))
        if H_li is not None:
            if H_li.shape[0] != H_perfect.shape[0] and H_li.ndim == 3:
                H_li = H_li.T
            if H_li.ndim == 3 and H_li.shape[1] == 14 and H_li.shape[2] == 132:
                H_li = np.transpose(H_li, (0, 2, 1))

    return H_perfect, H_in, pilot_rows, pilot_cols, H_perfect_ori, H_li, mat_dict


def split_indices(N: int, train_frac: float = 0.70, val_frac: float = 0.15, seed: int = 1234):
    """Return reproducible (train, val, test) index arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


# =============================================================================
# 6. EVALUATION METRICS
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
# 7. TRAINING STEP IMPLEMENTATIONS
# =============================================================================
@tf.function
def _train_step_coral(model, x_src_sc, y_src_sc, x_tgt_sc, optimizer,
                      est_weight=1.0, domain_weight=0.5, ssim_weight=0.1,
                      lower_range=-1, standardize=False, selected_layers=None):
    """
    Combined Supervised Source Estimation + Unsupervised Target CORAL Adaptation step.
    """
    with tf.GradientTape() as tape:
        # 1. Source forward pass with layer extraction
        y_src_pred_sc, src_features = model(
            x_src_sc, training=True, return_features=True, selected_layers=selected_layers
        )
        
        # Supervised Source Estimation Loss
        mse_loss = tf.reduce_mean(tf.square(y_src_pred_sc - y_src_sc))
        if ssim_weight > 0.0:
            if standardize:
                max_val = tf.maximum(tf.reduce_max(y_src_sc) - tf.reduce_min(y_src_sc), 1e-8)
            else:
                max_val = tf.cast(2.0 if lower_range == -1 else 1.0, tf.float32)
            ssim_val = tf_ssim(y_src_sc, y_src_pred_sc, max_val=max_val)
            ssim_loss = tf.reduce_mean(1.0 - ssim_val)
            loss_est = (1.0 - ssim_weight) * mse_loss + ssim_weight * ssim_loss
        else:
            ssim_loss = tf.constant(0.0, dtype=tf.float32)
            loss_est = mse_loss

        # 2. Target forward pass with layer extraction
        _, tgt_features = model(
            x_tgt_sc, training=True, return_features=True, selected_layers=selected_layers
        )
        
        # 3. CORAL domain alignment loss across selected layers
        coral_loss = compute_coral_loss(src_features, tgt_features)
        
        # Total Weighted Loss
        total_loss = est_weight * loss_est + domain_weight * coral_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, mse_loss, ssim_loss, coral_loss


@tf.function
def _train_step_source_only(model, x_src_sc, y_src_sc, optimizer,
                            est_weight=1.0, ssim_weight=0.1, lower_range=-1, standardize=False):
    """
    Supervised Source-Only Training step (No target domain adaptation).
    """
    with tf.GradientTape() as tape:
        y_src_pred_sc = model(x_src_sc, training=True, return_features=False)
        mse_loss = tf.reduce_mean(tf.square(y_src_pred_sc - y_src_sc))
        if ssim_weight > 0.0:
            if standardize:
                max_val = tf.maximum(tf.reduce_max(y_src_sc) - tf.reduce_min(y_src_sc), 1e-8)
            else:
                max_val = tf.cast(2.0 if lower_range == -1 else 1.0, tf.float32)
            ssim_val = tf_ssim(y_src_sc, y_src_pred_sc, max_val=max_val)
            ssim_loss = tf.reduce_mean(1.0 - ssim_val)
            loss_est = (1.0 - ssim_weight) * mse_loss + ssim_weight * ssim_loss
        else:
            ssim_loss = tf.constant(0.0, dtype=tf.float32)
            loss_est = mse_loss

        total_loss = est_weight * loss_est
        coral_loss = tf.constant(0.0, dtype=tf.float32)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, mse_loss, ssim_loss, coral_loss


def infer_full_dataset(model, H_perf_data, H_in_pilots, batch_size=16, lower_range=-1, standardize=False):
    """
    Run batched inference over dataset and return reconstructed complex grids.
    """
    N_samples = H_perf_data.shape[0]
    preds = []
    for i in range(0, N_samples, batch_size):
        idx = slice(i, min(i + batch_size, N_samples))
        x_sc, y_sc, v1, v2 = preprocess_batch(H_perf_data[idx], H_in_pilots[idx], lower_range, standardize)
        y_pred_sc = model(x_sc, training=False, return_features=False)
        if standardize:
            y_pred = deStandardize_ha02(y_pred_sc, v1, v2)
        else:
            y_pred = deMinMax_ha02(y_pred_sc, v1, v2, lower_range)
        y_pred_np = y_pred.numpy()
        H_comp = y_pred_np[..., 0] + 1j * y_pred_np[..., 1]
        preds.append(H_comp)
    return np.concatenate(preds, axis=0) if preds else np.empty((0,))


def extract_dataset_features(model, H_perf_data, H_in_pilots, batch_size=16, lower_range=-1, standardize=False, selected_layers=None):
    """
    Extract intermediate layer representations across dataset.
    Returns dict mapping layer_name (e.g. 'layer1') -> np.ndarray of shape (N, feature_dim).
    """
    if selected_layers is None or len(selected_layers) == 0:
        return {}
    N_samples = H_perf_data.shape[0]
    layer_feats = {lyr: [] for lyr in selected_layers}
    for i in range(0, N_samples, batch_size):
        idx = slice(i, min(i + batch_size, N_samples))
        x_sc, _, _, _ = preprocess_batch(H_perf_data[idx], H_in_pilots[idx], lower_range, standardize)
        _, feats = model(x_sc, training=False, return_features=True, selected_layers=selected_layers)
        for lyr, f_t in zip(selected_layers, feats):
            layer_feats[lyr].append(f_t.numpy())
    return {lyr: np.concatenate(layer_feats[lyr], axis=0) if layer_feats[lyr] else np.empty((0,)) for lyr in selected_layers}


def save_test_channel_mat(save_filepath, H_perf_test, H_perf_ori_test, H_ls_test, H_pred_test,
                          pilot_rows, pilot_cols, H_li_test, test_indices, snr_str, model_type_tag):
    """Export test channel grids to MAT format matching OpenNTN benchmark format."""
    test_dict = {
        'H_perfect_test': H_perf_test,
        'H_original_test': H_perf_ori_test if H_perf_ori_test is not None else H_perf_test,
        'H_LS_test': H_ls_test,
        'H_output_test': H_pred_test,
        'pilot_rows': pilot_rows + 1 if np.min(pilot_rows) == 0 else pilot_rows,
        'pilot_cols': pilot_cols + 1 if np.min(pilot_cols) == 0 else pilot_cols,
        'test_indices': test_indices,
        'snr': snr_str,
        'model_type': model_type_tag
    }
    if H_li_test is not None:
        test_dict['H_LI_test'] = H_li_test
    savemat(save_filepath, test_dict)
    print(f"[Save] Exported test MAT file -> {save_filepath}")


# =============================================================================
# 8. MAIN SCRIPT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="HA02 Attention Model Domain Adaptation with Configurable Multi/Single-Layer CORAL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--source-dir', type=str, default=SOURCE_DIR, help="Source dataset directory")
    parser.add_argument('--target-dir', type=str, default=TARGET_DIR, help="Target dataset directory")
    parser.add_argument('--save-dir', type=str, default=DEFAULT_SAVE_DIR, help="Folder directory to save results (defaults to './results')")
    parser.add_argument('--snr', type=int, default=DEFAULT_SNR, help="Channel SNR in dB (e.g. -15, -10, -5, 0, 5, 10, 15)")
    parser.add_argument('--type', type=str, default=MODEL_TYPE, choices=['LS', 'LI', 'Prac', 'ls', 'li', 'prac'], help="Model input type")
    parser.add_argument('--only-source', action='store_true', default=ONLY_SOURCE, help="Train using source-only data (no CORAL)")
    
    # Layer extraction parameter: supports multiple or single layers
    parser.add_argument(
        '--coral-layers', 
        nargs='+', 
        default=CORAL_LAYERS,
        help="Layers to extract for CORAL loss alignment (e.g., '--coral-layers layer1 layer2' for multi-layer, or '--coral-layers layer1' for single-layer). Available: layer1, layer2, layer3"
    )
    
    parser.add_argument('--domain-weight', type=float, default=0.5, help="CORAL domain loss weight (lambda)")
    parser.add_argument('--est-weight', type=float, default=1.0, help="Estimation loss weight")
    parser.add_argument('--ssim-weight', type=float, default=0.1, help="SSIM loss weight (0.0 to disable)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument('--n-epochs', type=int, default=N_EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('--train-frac', type=float, default=DEFAULT_TRAIN_FRAC, help="Train split fraction")
    parser.add_argument('--val-frac', type=float, default=DEFAULT_VAL_FRAC, help="Validation split fraction")
    parser.add_argument('--norm-approach', type=str, default='minmax', choices=['minmax', 'std'], help="Normalization approach")
    parser.add_argument('--lower-range', type=int, default=-1, choices=[0, -1], help="Scaling range for minmax")
    parser.add_argument('--test-code', action='store_true', default=TEST_CODE, help="Run with subset of data for testing")
    parser.add_argument('--save-features', action='store_true', default=SAVE_FEATURES, help="Extract and save intermediate features at begin, mid, and last epoch to .mat")
    parser.add_argument('--save-model', action='store_true', help="Save trained model checkpoints and weights")
    parser.add_argument('--no-gpu', action='store_true', help="Disable GPU execution")

    args = parser.parse_args()

    # GPU configuration
    if args.no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass

    # Normalize coral-layers (handle list of strings or comma-separated string)
    selected_layers = []
    for item in args.coral_layers:
        for sub_item in item.split(','):
            sub_clean = sub_item.strip().lower()
            if sub_clean:
                selected_layers.append(sub_clean)
    
    valid_layers = ['layer1', 'layer2', 'layer3']
    for lyr in selected_layers:
        if lyr not in valid_layers:
            raise ValueError(f"Invalid layer '{lyr}' in --coral-layers. Valid options are: {valid_layers}")

    is_standardize = (args.norm_approach == 'std')
    domain_weight = 0.0 if args.only_source else args.domain_weight

    # Resolve dataset paths
    source_mat_path = get_mat_file(args.source_dir, args.snr)
    target_mat_path = get_mat_file(args.target_dir, args.snr)

    print("=" * 80)
    print(f"HA02 Attention Domain Adaptation | Mode: {'Source-Only' if args.only_source else 'CORAL UDA'}")
    if not args.only_source:
        print(f"CORAL Extracted Layers: {selected_layers} ({'Multi-layer' if len(selected_layers) > 1 else 'Single-layer'})")
        print(f"CORAL Loss Weight (lambda): {domain_weight}")
    print(f"Source Dataset: {source_mat_path}")
    print(f"Target Dataset: {target_mat_path}")
    print(f"SNR: {args.snr} dB | Input Type: {args.type} | Normalization: {args.norm_approach} (range: [{args.lower_range}, 1])")
    print("=" * 80)

    # Save Directory Setup (saves directly to args.save_dir or ./results/ inside current directory)
    if args.save_dir and args.save_dir.strip():
        output_dir = os.path.abspath(args.save_dir)
    else:
        output_dir = os.path.join(current_dir, 'results')

    os.makedirs(output_dir, exist_ok=True)
    print(f"Experiment results will be saved to: {output_dir}")

    # ============ Load Datasets ============
    H_perf_src, H_in_src, p_rows, p_cols, H_perf_ori_src, H_li_src, _ = load_dataset(source_mat_path, args.type)
    H_perf_tgt, H_in_tgt, _, _, H_perf_ori_tgt, H_li_tgt, _ = load_dataset(target_mat_path, args.type)

    N_src = H_perf_src.shape[0]
    N_tgt = H_perf_tgt.shape[0]
    print(f"Loaded Source samples: {N_src} | Target samples: {N_tgt}")
    print(f"Pilot Element Shape: {H_in_src.shape[1:]} | Full Grid Shape: {H_perf_src.shape[1:]}")

    # Split Source and Target into Train / Val / Test sets
    if args.test_code:
        idx_train_src, idx_val_src, idx_test_src = np.arange(0, 64), np.arange(64, 80), np.arange(80, 96)
        idx_train_tgt, idx_val_tgt, idx_test_tgt = np.arange(0, 64), np.arange(64, 80), np.arange(80, 96)
        args.n_epochs = 5
    else:
        idx_train_src, idx_val_src, idx_test_src = split_indices(N_src, args.train_frac, args.val_frac, seed=1234)
        idx_train_tgt, idx_val_tgt, idx_test_tgt = split_indices(N_tgt, args.train_frac, args.val_frac, seed=1234)

    # Align training samples to batch size
    n_train = min(len(idx_train_src), len(idx_train_tgt)) // args.batch_size * args.batch_size
    idx_train_src = idx_train_src[:n_train]
    idx_train_tgt = idx_train_tgt[:n_train]

    print(f"Dataset Partition -> Train: {n_train} | Val: (Src: {len(idx_val_src)}, Tgt: {len(idx_val_tgt)}) | Test: (Src: {len(idx_test_src)}, Tgt: {len(idx_test_tgt)})")

    # ============ Initialize Model & Optimizer ============
    num_pilot_elems = H_in_src.shape[1]  # 88
    total_grid_elems = H_perf_src.shape[1] * H_perf_src.shape[2]  # 132 * 14 = 1848

    model = HA02Model(
        num_pilot_elems=num_pilot_elems,
        total_grid_elems=total_grid_elems,
        num_channels=2,
        num_heads=2,
        n_filter=2
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Compile a dummy forward pass to build variables
    dummy_input = tf.zeros((1, num_pilot_elems, 2), dtype=tf.float32)
    _ = model(dummy_input, training=False, return_features=True, selected_layers=selected_layers)
    print(f"Initialized HA02Model with {model.count_params():,} parameters.")

    # Tracking History
    history = {
        'train_loss': [],
        'train_mse': [],
        'train_ssim': [],
        'train_coral': [],
        'val_loss': [],
        'val_mse_src': [],
        'val_mse_tgt': [],
        'val_nmse_db_src': [],
        'val_nmse_db_tgt': [],
        'val_ssim_src': [],
        'val_ssim_tgt': [],
        'val_coral': [],
        'selected_layers': selected_layers
    }

    saved_features = {}
    mid_epoch = args.n_epochs // 2
    feature_checkpoint_epochs = {
        0: 'begin',
        mid_epoch: 'mid',
        args.n_epochs - 1: 'last'
    }

    start_time = time.perf_counter()

    # ============ Main Training Loop ============
    print("\nStarting Training ...")
    num_batches = n_train // args.batch_size

    for epoch in range(args.n_epochs):
        ep_start = time.perf_counter()
        
        # Shuffle train indices each epoch
        perm_src = np.random.permutation(n_train)
        perm_tgt = np.random.permutation(n_train)
        
        ep_train_loss = 0.0
        ep_train_mse = 0.0
        ep_train_ssim = 0.0
        ep_train_coral = 0.0

        for b in range(num_batches):
            b_src = idx_train_src[perm_src[b * args.batch_size : (b + 1) * args.batch_size]]
            b_tgt = idx_train_tgt[perm_tgt[b * args.batch_size : (b + 1) * args.batch_size]]

            x_src_sc, y_src_sc, _, _ = preprocess_batch(
                H_perf_src[b_src], H_in_src[b_src], args.lower_range, is_standardize
            )
            x_tgt_sc, y_tgt_sc, _, _ = preprocess_batch(
                H_perf_tgt[b_tgt], H_in_tgt[b_tgt], args.lower_range, is_standardize
            )

            if args.only_source:
                t_loss, mse_l, ssim_l, coral_l = _train_step_source_only(
                    model, x_src_sc, y_src_sc, optimizer,
                    est_weight=args.est_weight, ssim_weight=args.ssim_weight,
                    lower_range=args.lower_range, standardize=is_standardize
                )
            else:
                t_loss, mse_l, ssim_l, coral_l = _train_step_coral(
                    model, x_src_sc, y_src_sc, x_tgt_sc, optimizer,
                    est_weight=args.est_weight, domain_weight=domain_weight,
                    ssim_weight=args.ssim_weight, lower_range=args.lower_range,
                    standardize=is_standardize, selected_layers=selected_layers
                )

            ep_train_loss += float(t_loss)
            ep_train_mse += float(mse_l)
            ep_train_ssim += float(ssim_l)
            ep_train_coral += float(coral_l)

        # Average training epoch metrics
        ep_train_loss /= num_batches
        ep_train_mse /= num_batches
        ep_train_ssim /= num_batches
        ep_train_coral /= num_batches

        # ============ Validation Step ============
        val_pred_src = infer_full_dataset(model, H_perf_src[idx_val_src], H_in_src[idx_val_src], args.batch_size, args.lower_range, is_standardize)
        val_pred_tgt = infer_full_dataset(model, H_perf_tgt[idx_val_tgt], H_in_tgt[idx_val_tgt], args.batch_size, args.lower_range, is_standardize)

        val_nmse_src = compute_nmse(val_pred_src, H_perf_src[idx_val_src])
        val_nmse_tgt = compute_nmse(val_pred_tgt, H_perf_tgt[idx_val_tgt])
        val_nmse_db_src = compute_nmse_db(val_pred_src, H_perf_src[idx_val_src])
        val_nmse_db_tgt = compute_nmse_db(val_pred_tgt, H_perf_tgt[idx_val_tgt])
        val_mmse_src = compute_mmse(val_pred_src, H_perf_src[idx_val_src])
        val_mmse_tgt = compute_mmse(val_pred_tgt, H_perf_tgt[idx_val_tgt])
        val_ssim_src = compute_ssim_batch(val_pred_src, H_perf_src[idx_val_src])
        val_ssim_tgt = compute_ssim_batch(val_pred_tgt, H_perf_tgt[idx_val_tgt])

        # Record metrics
        history['train_loss'].append(ep_train_loss)
        history['train_mse'].append(ep_train_mse)
        history['train_ssim'].append(ep_train_ssim)
        history['train_coral'].append(ep_train_coral)
        history['val_loss'].append(val_mmse_src)
        history['val_mse_src'].append(val_mmse_src)
        history['val_mse_tgt'].append(val_mmse_tgt)
        history['val_nmse_db_src'].append(val_nmse_db_src)
        history['val_nmse_db_tgt'].append(val_nmse_db_tgt)
        history['val_ssim_src'].append(val_ssim_src)
        history['val_ssim_tgt'].append(val_ssim_tgt)
        history['val_coral'].append(ep_train_coral)

        # ============ Extract Features at Specified Epochs ============
        if args.save_features and (epoch in feature_checkpoint_epochs):
            stage = feature_checkpoint_epochs[epoch]
            print(f"  [Feature Extraction] Extracting intermediate features at '{stage}' stage (Epoch {epoch+1}/{args.n_epochs})...")
            
            src_f = extract_dataset_features(
                model, H_perf_src[idx_train_src], H_in_src[idx_train_src],
                batch_size=args.batch_size, lower_range=args.lower_range,
                standardize=is_standardize, selected_layers=selected_layers
            )
            tgt_f = extract_dataset_features(
                model, H_perf_tgt[idx_train_tgt], H_in_tgt[idx_train_tgt],
                batch_size=args.batch_size, lower_range=args.lower_range,
                standardize=is_standardize, selected_layers=selected_layers
            )
            
            for lyr in selected_layers:
                alias = lyr.replace('layer', 'l')  # e.g. 'l1', 'l2', 'l3'
                
                # Primary keys matching format (e.g. features_begin_l1, features_mid_l1, features_last_l1)
                saved_features[f"features_{stage}_{alias}"] = src_f[lyr]
                saved_features[f"features_{stage}_{alias}_src"] = src_f[lyr]
                saved_features[f"src_features_{stage}_{alias}"] = src_f[lyr]
                
                if not args.only_source:
                    saved_features[f"features_{stage}_{alias}_tgt"] = tgt_f[lyr]
                    saved_features[f"tgt_features_{stage}_{alias}"] = tgt_f[lyr]
                
                # Full name aliases (e.g. features_begin_layer1_src)
                saved_features[f"features_{stage}_{lyr}_src"] = src_f[lyr]
                if not args.only_source:
                    saved_features[f"features_{stage}_{lyr}_tgt"] = tgt_f[lyr]

        ep_duration = time.perf_counter() - ep_start
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.n_epochs - 1:
            print(f"Epoch [{epoch+1:03d}/{args.n_epochs:03d}] ({ep_duration:.1f}s) | "
                  f"Train Loss: {ep_train_loss:.5f} (MSE: {ep_train_mse:.5f}, CORAL: {ep_train_coral:.5f}) | "
                  f"Val NMSE (dB) -> Src: {val_nmse_db_src:.2f} dB, Tgt: {val_nmse_db_tgt:.2f} dB | "
                  f"Val SSIM -> Src: {val_ssim_src:.4f}, Tgt: {val_ssim_tgt:.4f}")

    total_time = time.perf_counter() - start_time
    print(f"\nTraining completed in {total_time/60.0:.2f} minutes.")

    # ============ Final Test Evaluation ============
    print("\nEvaluating on Held-Out Test Sets ...")
    test_pred_src = infer_full_dataset(model, H_perf_src[idx_test_src], H_in_src[idx_test_src], args.batch_size, args.lower_range, is_standardize)
    test_pred_tgt = infer_full_dataset(model, H_perf_tgt[idx_test_tgt], H_in_tgt[idx_test_tgt], args.batch_size, args.lower_range, is_standardize)

    test_nmse_db_src = compute_nmse_db(test_pred_src, H_perf_src[idx_test_src])
    test_nmse_db_tgt = compute_nmse_db(test_pred_tgt, H_perf_tgt[idx_test_tgt])
    test_mmse_src = compute_mmse(test_pred_src, H_perf_src[idx_test_src])
    test_mmse_tgt = compute_mmse(test_pred_tgt, H_perf_tgt[idx_test_tgt])
    test_ssim_src = compute_ssim_batch(test_pred_src, H_perf_src[idx_test_src])
    test_ssim_tgt = compute_ssim_batch(test_pred_tgt, H_perf_tgt[idx_test_tgt])

    print("=" * 70)
    print("                      FINAL TEST PERFORMANCE SUMMARY                  ")
    print("=" * 70)
    print(f"  Source Domain -> NMSE: {test_nmse_db_src:.2f} dB | MMSE: {test_mmse_src:.6e} | SSIM: {test_ssim_src:.4f}")
    print(f"  Target Domain -> NMSE: {test_nmse_db_tgt:.2f} dB | MMSE: {test_mmse_tgt:.6e} | SSIM: {test_ssim_tgt:.4f}")
    print("=" * 70)

    # Save Test MAT Files (Source and Target Domains)
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

    # Save final_epoch.txt report matching single_dataset schema
    txt_path = os.path.join(output_dir, 'final_epoch.txt')
    try:
        with open(txt_path, 'w') as f:
            f.write("=== FINAL EPOCH EVALUATION RESULTS ===\n")
            f.write(f"SNR (dB):             {args.snr}\n")
            f.write(f"Input Type:           {args.type}\n")
            f.write(f"Domain Adaptation:    {'Source-Only' if args.only_source else 'CORAL'}\n")
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

    # Save evaluation_results.mat matching single_dataset schema
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
        'snr': args.snr,
        'input_type': args.type,
        'coral_layers': np.array(selected_layers)
    }
    savemat(eval_path, eval_dict)
    print(f"[Save] Evaluation results -> {eval_path}")

    # Save History and Metrics MAT
    history_save_path = os.path.join(output_dir, 'training_history.mat')
    savemat(history_save_path, {k: np.array(v) for k, v in history.items()})
    print(f"[Save] Saved training history -> {history_save_path}")

    # Save Extracted Features MAT if requested
    if args.save_features and saved_features:
        feat_save_path = os.path.join(output_dir, 'extracted_features.mat')
        saved_features['selected_layers'] = np.array(selected_layers)
        saved_features['train_indices_src'] = idx_train_src
        saved_features['train_indices_tgt'] = idx_train_tgt
        savemat(feat_save_path, saved_features, do_compression=True)
        print(f"[Save] Exported extracted features ({', '.join([k for k in saved_features.keys() if 'features_' in k and k.endswith(('_l1', '_l2', '_l3'))])}) -> {feat_save_path}")

    # Export PDF Loss Plots
    try:
        import matplotlib.pyplot as plt
        
        # Loss Curves
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history['train_loss'], label='Train Total Loss', color='royalblue')
        ax.plot(history['train_mse'], label='Train MSE', color='seagreen')
        if not args.only_source:
            ax.plot(history['train_coral'], label='Train CORAL', color='darkorange')
        ax.plot(history['val_loss'], label='Val MSE (Src)', color='crimson', linestyle='--')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'HA02 CORAL Training Loss History ({", ".join(selected_layers)})', fontsize=13)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_total.pdf'), format='pdf')
        plt.close(fig)

        # NMSE Curves
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history['val_nmse_db_src'], label='Val NMSE (Source)', color='royalblue')
        ax.plot(history['val_nmse_db_tgt'], label='Val NMSE (Target)', color='crimson')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('NMSE (dB)', fontsize=12)
        ax.set_title('Validation NMSE History (dB)', fontsize=13)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'val_nmse_db.pdf'), format='pdf')
        plt.close(fig)

        # Visual Heatmaps of Sample 1
        if len(idx_test_tgt) > 0:
            sample_idx = 0
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            
            # 1. Perfect Reference
            im0 = axes[0].imshow(H_perf_tgt[idx_test_tgt[sample_idx]].real, aspect='auto', cmap='viridis')
            axes[0].set_title('Target Perfect Channel (Real)', fontsize=12)
            fig.colorbar(im0, ax=axes[0])
            
            # 2. Sparse Pilot Input Grid
            sparse_grid = np.zeros_like(H_perf_tgt[idx_test_tgt[sample_idx]])
            try:
                sparse_grid[p_rows, p_cols] = H_in_tgt[idx_test_tgt[sample_idx]]
            except Exception:
                sparse_grid[p_cols, p_rows] = H_in_tgt[idx_test_tgt[sample_idx]]
            im1 = axes[1].imshow(sparse_grid.real, aspect='auto', cmap='viridis')
            axes[1].set_title(f'Target Input Pilots H_{args.type} (Real)', fontsize=12)
            fig.colorbar(im1, ax=axes[1])
            
            # 3. Model Prediction
            im2 = axes[2].imshow(test_pred_tgt[sample_idx].real, aspect='auto', cmap='viridis')
            axes[2].set_title('HA02 Predicted Channel (Real)', fontsize=12)
            fig.colorbar(im2, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'sample_channel_reconstruction.pdf'), format='pdf')
            plt.close(fig)

        print(f"[PDF Export] Successfully exported loss curves and channel heatmaps to: {output_dir}")
    except Exception as e:
        print(f"[PDF Warning] Plot generation failed: {e}")

    # Save Model Weights if requested
    if args.save_model:
        weights_path = os.path.join(output_dir, 'ha02_model.weights.h5')
        try:
            model.save_weights(weights_path)
            print(f"[Model Save] Successfully saved model weights to: {weights_path}")
        except Exception as e:
            print(f"[Model Save Warning] Failed to save weights: {e}")

    print("\nCORAL Domain Adaptation pipeline finished successfully!")


if __name__ == '__main__':
    main()
