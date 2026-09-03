"""
====================================================================================================
CORAL DnCNN Domain Adaptation for NTN Channel Estimation (OpenNTN & MATLAB) - High-Performance
====================================================================================================

Overview
--------
This script trains and evaluates a Deep Residual Convolutional Neural Network (DnCNN / CNNGenerator)
with Correlation Alignment (CORAL) Unsupervised Domain Adaptation (UDA) for 5G Non-Terrestrial Network 
(NTN) channel estimation.

Domain shift (e.g., between different user velocities, propagation delays, or TDL profiles) is mitigated
using CORAL loss, which aligns the second-order statistics (covariance matrices) of intermediate feature 
representations between the source and target domains.

Spatial Global Average Pooling (GAP) for 2D Conv Features:
----------------------------------------------------------
Because SameShapeBlocks preserve spatial resolution (132 x 14 = 1,848 grid points), intermediate feature 
maps have shape [Batch, 132, 14, Channels]. Straight flattening would yield 946,176 dimensions per sample,
which would cause an out-of-memory crash during covariance calculation.
Therefore, Global Average Pooling (GAP) is applied across spatial axes [1, 2] to reduce each block's 
representation to [Batch, Channels] before computing the covariance alignment loss.

Channel Grid & Metric Tracking Features:
----------------------------------------
1. Channel Grid Observation (Input/LI, Output, Ground Truth):
   - Generates individual side-by-side PDFs:
     * `recon_source_train.pdf`, `recon_source_test.pdf`
     * `recon_target_train.pdf`, `recon_target_test.pdf`
     * `channel_reconstructions_all.pdf` (4-row consolidated comparison)
   - Saves plotted sample channel grids into `sample_reconstructions.mat` for easy replotting.
2. Full Metrics Tracking Across Epochs:
   - Tracks Total Loss, Source Estimation Loss, and CORAL Loss.
   - Tracks NMSE (dB), MSE, and SSIM on Source Train, Source Val, Target Train, and Target Val.
   - Generates PDF plots:
     * `loss_total.pdf`
     * `metrics_nmse_db.pdf`
     * `metrics_mse.pdf`
     * `metrics_ssim.pdf`
     * `metrics_summary_2x2.pdf`
   - Saves all epoch trajectories into `training_history.mat`.

Usage Examples
--------------
    # CORAL UDA on MATLAB A100 vs OpenNTN at SNR = 5 dB with layer alignment on block_2 and block_3
    python train_CORAL_DnCNN.py --source-dir A100_2p18e9_600km_70deg_30kHz --target-dir DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps --snr 5 --coral-layers block_2 block_3 --domain-weight 0.5 --save-features

    # Quick test run (small subset, 5 epochs)
    python train_CORAL_DnCNN.py --snr 5 --test-code
====================================================================================================
"""

import os
import sys
import time
import argparse

# NumPy 2.x compatibility monkey-patch
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

import h5py
import scipy.io
from scipy.io import savemat, loadmat
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
DEFAULT_SAVE_DIR    = ""           # Output save directory (defaults to './results' inside script directory)
DEFAULT_SNR         = 5            # dB
DEFAULT_INPUT_TYPE  = 'li'         # 'li', 'prac', or 'ls'
DEFAULT_EPOCHS      = 200
DEFAULT_BATCH_SIZE  = 16
DEFAULT_LR          = 1e-4
DEFAULT_TRAIN_FRAC  = 0.70
DEFAULT_VAL_FRAC    = 0.15
DEFAULT_LOWER_RANGE = -1           # minmax scaling range: -1 -> [-1, 1]
DEFAULT_SSIM_START  = 0.95         # Initial SSIM weight (MSE weight = 1 - w)
DEFAULT_SSIM_END    = 0.05         # Final SSIM weight
DEFAULT_DOMAIN_WEIGHT = 0.5        # CORAL loss penalty (lambda)
DEFAULT_CORAL_LAYERS = ['block_2', 'block_3']
DEFAULT_PROJ_DIM     = 64           # Projected embedding dimension for CORAL alignment
DEFAULT_N_BLOCKS     = 4
DEFAULT_CLIP_EXTRAP  = False
# ============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
for p in [project_root,
          os.path.join(project_root, 'Domain_Adversarial', 'helper'),
          os.path.join(project_root, 'JMMD', 'helper')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils_GAN import CNNGenerator                         # JMMD/helper/utils_GAN.py


def deMinMax(x_normd, x_min, x_max, lower_range=-1):
    diff = x_max - x_min
    if lower_range == -1:
        return (x_normd + 1.0) * diff / 2.0 + x_min
    return x_normd * diff + x_min


def deStandardize(x_scaled, mean_x, std_x):
    return x_scaled * std_x + mean_x


def compute_nmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    err = np.sum(np.abs(y_pred - y_true) ** 2)
    ref = np.sum(np.abs(y_true) ** 2)
    return float(err / (ref + 1e-30))


def compute_mmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true) ** 2))


def compute_ssim_batch(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    yp = np.stack([y_pred.real, y_pred.imag], axis=-1).astype(np.float32)
    yt = np.stack([y_true.real, y_true.imag], axis=-1).astype(np.float32)
    val = tf_ssim(tf.convert_to_tensor(yt), tf.convert_to_tensor(yp), max_val=2.0)
    return float(tf.reduce_mean(val).numpy())


# ============================================================================
# 1. DATASET RESOLUTION & ADAPTIVE LOADING
# ============================================================================

def find_any_mat_file(base_dir: str) -> str:
    """Recursively search base_dir for the first valid channel .mat file."""
    if not os.path.exists(base_dir):
        return None
    for root, _, files in os.walk(base_dir):
        for f in sorted(files):
            if f.endswith('.mat') and not f.startswith(('inferredChannel', 'testChannel', 'training_history', 'extracted_features', 'synthesized_results', 'sample_reconstructions')):
                return os.path.join(root, f)
    return None


def get_mat_file(data_root: str, snr: int = 5) -> str:
    """
    Robustly locate the .mat data file for the requested SNR, supporting:
    - SNR folder variations: 'SNR_-10dB', '-10dB', 'SNR_-10', '-10', '5dB', etc.
    - Relative paths from workspace root or script directory
    - Scenario name substring matching (in generatedChan/MATLAB or generatedChan/OpenNTN)
    """
    if data_root and os.path.isfile(data_root) and data_root.endswith('.mat'):
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
        os.path.join(project_root, 'generatedChan', 'MATLAB', 'A100_2p18e9_600km_70deg_30kHz'),
        os.path.join(project_root, 'generatedChan', 'OpenNTN', 'DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps'),
        os.path.join(project_root, 'generatedChan', 'OpenNTN', 'DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps')
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
        for snr_var in snr_variations:
            snr_dir = os.path.join(root, snr_var)
            if os.path.isdir(snr_dir):
                mat_file = find_any_mat_file(snr_dir)
                if mat_file:
                    return os.path.abspath(mat_file)
        mat_file = find_any_mat_file(root)
        if mat_file:
            return os.path.abspath(mat_file)

    raise FileNotFoundError(
        f"Could not find any .mat data files for SNR={snr} in any searched location.\n"
        f"Searched roots: {candidate_roots}"
    )


def load_dataset_dncnn(mat_filepath: str, input_type: str = 'li') -> dict:
    """
    Load channel data from MATLAB v7.3 (HDF5) or legacy v7 format.
    Returns dictionary with channel arrays formatted into [N, 132, 14] complex.
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

    # Convert 1-based MATLAB coordinates to 0-based Python coordinates
    p_cols = np.squeeze(mat_dict['pilot_cols']).astype(int) - 1
    p_rows = np.squeeze(mat_dict['pilot_rows']).astype(int) - 1
    mat_dict['pilot_cols'] = p_cols
    mat_dict['pilot_rows'] = p_rows

    # Resolve H_input
    if input_type in ['ls', 'ls_ori']:
        ls_key = 'H_ls_pilots_ori' if input_type == 'ls_ori' and 'H_ls_pilots_ori' in mat_dict else 'H_ls_pilots'
        if ls_key not in mat_dict:
            for alt in ['H_ls_pilots', 'H_ls_pilots_ori', 'H_LS_comp', 'H_LS_full']:
                if alt in mat_dict:
                    ls_key = alt
                    break
        H_pilots = mat_dict[ls_key]
        if H_pilots.ndim == 2 and H_pilots.shape[0] != len(H_perfect) and H_pilots.shape[1] == len(H_perfect):
            H_pilots = H_pilots.T
        N_s = len(H_perfect)
        H_in = np.zeros((N_s, 132, 14), dtype=np.complex64)
        for i in range(N_s):
            H_in[i, p_rows, p_cols] = H_pilots[i, :]
    else:
        input_key_map = {'prac': 'H_prac', 'li': 'H_li', 'li_ori': 'H_li_ori'}
        target_key = input_key_map.get(input_type, 'H_li')
        if target_key not in mat_dict or mat_dict[target_key] is None:
            for alt in [target_key, 'H_li', 'H_li_ori', 'H_prac', 'H_perfect']:
                if alt in mat_dict and mat_dict[alt] is not None:
                    target_key = alt
                    break
        H_in = format_3d(mat_dict[target_key])

    mat_dict[f'H_{input_type}'] = H_in
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


# ============================================================================
# 2. DATA PREPROCESSING & BATCHING
# ============================================================================

def preprocess_batch(H_perf: np.ndarray, H_in: np.ndarray, lower_range: int = -1,
                     clip_extrap: bool = False, pilot_bounds: tuple = None,
                     standardize: bool = False):
    """Normalize complex channel arrays to 2-channel real tensors [B, 132, 14, 2]."""
    x_real = np.stack([H_in.real, H_in.imag], axis=-1).astype(np.float32)
    y_real = np.stack([H_perf.real, H_perf.imag], axis=-1).astype(np.float32)

    if clip_extrap and pilot_bounds is not None:
        r_min, r_max, c_min, c_max = pilot_bounds
        p_region = x_real[:, r_min:r_max, c_min:c_max, :]
        p_min = np.min(p_region, axis=(1, 2), keepdims=True)
        p_max = np.max(p_region, axis=(1, 2), keepdims=True)
        x_real = np.clip(x_real, p_min, p_max)

    if standardize:
        mean_x = np.mean(x_real, axis=(1, 2), keepdims=True)
        std_x = np.std(x_real, axis=(1, 2), keepdims=True) + 1e-8
        x_scaled = (x_real - mean_x) / std_x
        y_scaled = (y_real - mean_x) / std_x
        return tf.convert_to_tensor(x_scaled, dtype=tf.float32), tf.convert_to_tensor(y_scaled, dtype=tf.float32), mean_x, std_x
    else:
        min_x = np.min(x_real, axis=(1, 2), keepdims=True)
        max_x = np.max(x_real, axis=(1, 2), keepdims=True)
        diff = np.maximum(max_x - min_x, 1e-8)
        if lower_range == -1:
            x_scaled = 2.0 * (x_real - min_x) / diff - 1.0
            y_scaled = 2.0 * (y_real - min_x) / diff - 1.0
        else:
            x_scaled = (x_real - min_x) / diff
            y_scaled = (y_real - min_x) / diff
        return tf.convert_to_tensor(x_scaled, dtype=tf.float32), tf.convert_to_tensor(y_scaled, dtype=tf.float32), min_x, max_x

# ============================================================================
# 3. PROJECTION HEAD NETWORK & PROJECTED CORAL LOSS
# ============================================================================

class ProjectionHead(tf.keras.layers.Layer):
    """
    Dedicated Non-Linear Projection Head Network.
    Applies spatial Global Average Pooling (GAP) to 4D CNN tensors [B, 132, 14, C] -> [B, C],
    then maps through a 2-layer MLP with LayerNorm and GeLU non-linearity:
    [B, C] -> Dense(hidden_dim) -> LayerNorm -> GeLU -> Dense(proj_dim) -> [B, proj_dim].
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, proj_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.dense1 = tf.keras.layers.Dense(hidden_dim, kernel_initializer='glorot_uniform', name="dense_proj1")
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln_proj")
        self.act = tf.keras.layers.Activation('gelu', name="act_proj")
        self.dense2 = tf.keras.layers.Dense(proj_dim, kernel_initializer='glorot_uniform', name="dense_proj2")

    def call(self, x, training=False):
        # 1. Spatial Global Average Pooling across (132, 14) grid for 4D feature maps
        if len(x.shape) == 4:
            x = tf.reduce_mean(x, axis=[1, 2])  # [B, C]
        elif len(x.shape) > 2:
            x = tf.reshape(x, [tf.shape(x)[0], -1])

        h = self.dense1(x)
        h = self.norm(h, training=training)
        h = self.act(h)
        p = self.dense2(h)
        return p


def get_block_filters(block_idx: int, n_blocks: int) -> int:
    """Calculate the filter channels of block_idx in CNNGenerator pyramid."""
    if block_idx == 0 or block_idx == n_blocks - 1:
        return 64
    if block_idx < n_blocks // 2:
        return min(32 * (4 ** block_idx), 1024)
    mirror_idx = n_blocks - 1 - block_idx
    return min(32 * (4 ** mirror_idx), 1024)


def build_projection_heads(selected_layers: list, n_blocks: int = 6, custom_proj_dim: int = 64) -> dict:
    """Instantiate a dedicated non-linear projection head for each selected layer."""
    heads = {}
    for lyr in selected_layers:
        if 'block_' in lyr:
            try:
                b_num = int(lyr.split('_')[-1])
                b_idx = b_num - 1
                in_dim = get_block_filters(b_idx, n_blocks)
            except Exception:
                in_dim = 128
        else:
            in_dim = 128
        hidden_dim = max(in_dim // 2, 128)
        heads[lyr] = ProjectionHead(in_dim=in_dim, hidden_dim=hidden_dim, proj_dim=custom_proj_dim)
    return heads


@tf.function
def coral_loss_single_layer(p_s: tf.Tensor, p_t: tf.Tensor) -> tf.Tensor:
    """
    Computes CORAL loss on projected embeddings P_src and P_tgt.
    Loss = ||Cov(P_src) - Cov(P_tgt)||_F^2 / (4 * d^2)
    """
    n_s = tf.cast(tf.shape(p_s)[0], tf.float32)
    n_t = tf.cast(tf.shape(p_t)[0], tf.float32)
    d = tf.cast(tf.shape(p_s)[1], tf.float32)

    p_s_centered = p_s - tf.reduce_mean(p_s, axis=0, keepdims=True)
    p_t_centered = p_t - tf.reduce_mean(p_t, axis=0, keepdims=True)

    cov_s = tf.matmul(p_s_centered, p_s_centered, transpose_a=True) / tf.maximum(n_s - 1.0, 1.0)
    cov_t = tf.matmul(p_t_centered, p_t_centered, transpose_a=True) / tf.maximum(n_t - 1.0, 1.0)

    cov_diff_sq = tf.reduce_sum(tf.square(cov_s - cov_t))
    return cov_diff_sq / (4.0 * d * d + 1e-12)


# ============================================================================
# 4. INFERENCE & FEATURE EXTRACTION HELPERS
# ============================================================================

def infer_channel(model: CNNGenerator, H_perf: np.ndarray, H_in: np.ndarray,
                  batch_size: int, lower_range: int = -1, clip_extrap: bool = False,
                  pilot_bounds: tuple = None, standardize: bool = False) -> np.ndarray:
    """Run model inference over dataset and return complex array [N, 132, 14]."""
    N = H_perf.shape[0]
    out = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_sc, _, v1, v2 = preprocess_batch(H_perf[start:end], H_in[start:end], lower_range,
                                           clip_extrap=clip_extrap, pilot_bounds=pilot_bounds,
                                           standardize=standardize)
        residual, _ = model(x_sc, training=False)
        x_corr = x_sc + residual
        if standardize:
            x_denorm = deStandardize(x_corr, v1, v2)
        else:
            x_denorm = deMinMax(x_corr, v1, v2, lower_range=lower_range)
        x_np = x_denorm.numpy()
        out.append(x_np[..., 0] + 1j * x_np[..., 1])

    return np.concatenate(out, axis=0) if len(out) > 0 else np.empty((0, 132, 14), dtype=np.complex64)


def extract_features_and_projections(model: CNNGenerator, proj_heads: dict, H_perf: np.ndarray, H_in: np.ndarray,
                                     batch_size: int, selected_layers: list, lower_range: int = -1,
                                     clip_extrap: bool = False, pilot_bounds: tuple = None,
                                     standardize: bool = False) -> tuple:
    """Extract intermediate features: raw (GAP-pooled [B, C]) and non-linear projected embeddings [B, d_proj]."""
    N = H_in.shape[0]
    raw_dict = {lyr: [] for lyr in selected_layers}
    proj_dict = {lyr: [] for lyr in selected_layers}
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_sc, _, _, _ = preprocess_batch(H_perf[start:end], H_in[start:end], lower_range,
                                         clip_extrap=clip_extrap, pilot_bounds=pilot_bounds,
                                         standardize=standardize)
        _, f_list = model(x_sc, training=False, return_features=True)
        for lyr_name, f_t in zip(selected_layers, f_list):
            if len(f_t.shape) == 4:
                f_pool = tf.reduce_mean(f_t, axis=[1, 2])
            else:
                f_pool = tf.reshape(f_t, [tf.shape(f_t)[0], -1])
            p_t = proj_heads[lyr_name](f_t, training=False)
            raw_dict[lyr_name].append(f_pool.numpy())
            proj_dict[lyr_name].append(p_t.numpy())

    return (
        {k: np.concatenate(v, axis=0) if len(v) > 0 else np.empty((0,)) for k, v in raw_dict.items()},
        {k: np.concatenate(v, axis=0) if len(v) > 0 else np.empty((0,)) for k, v in proj_dict.items()}
    )


def save_test_channel_mat(filepath: str, H_perf: np.ndarray, H_perf_ori: np.ndarray,
                          H_in: np.ndarray, H_pred: np.ndarray, mat_dict: dict,
                          indices: np.ndarray, snr: int, model_type: str):
    """Save test channel representations to MATLAB MAT file matching standard benchmark schema."""
    H_ls_candidate = None
    for k in ['H_ls_pilots', 'H_ls_pilots_ori']:
        if k in mat_dict and mat_dict[k] is not None:
            arr = mat_dict[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                n_total = len(H_perf) if H_perf is not None else len(mat_dict['H_perfect'])
                if arr.shape[0] == 88 and arr.shape[1] == n_total:
                    arr = arr.T
                if arr.shape[0] == n_total and arr.shape[1] == 88:
                    H_ls_candidate = arr[indices]
                    break

    if H_ls_candidate is None:
        p_r = mat_dict['pilot_rows']
        p_c = mat_dict['pilot_cols']
        src_grid = H_in if (H_in is not None and H_in.ndim == 3) else mat_dict['H_perfect'][indices]
        H_ls_candidate = src_grid[:, p_r, p_c] if src_grid.shape[1] == 132 else src_grid[:, p_c, p_r]

    out = {
        'H_perfect_test': H_perf,
        'H_original_test': H_perf_ori if H_perf_ori is not None else H_perf,
        'H_LS_test': H_ls_candidate,
        'H_output_test': H_pred,
        'pilot_rows': mat_dict['pilot_rows'] + 1,  # 1-indexed for MATLAB
        'pilot_cols': mat_dict['pilot_cols'] + 1,  # 1-indexed for MATLAB
        'test_indices': indices,
        'snr': snr,
        'model_type': model_type
    }
    if 'H_li' in mat_dict and mat_dict['H_li'] is not None:
        out['H_LI_test'] = mat_dict['H_li'][indices]

    savemat(filepath, out)
    print(f"[Save] Exported test MAT file -> {filepath}")


# ============================================================================
# 5. VISUALIZATION & PLOTTING HELPERS
# ============================================================================

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
    ax.set_title('DnCNN CORAL Training Loss Progression', fontsize=12, fontweight='bold')
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


def save_single_reconstruction_pdf(H_true, H_in, H_pred, title_str: str, out_filename: str, save_dir: str):
    """Generate side-by-side 2D heatmap PDF of Ground Truth, Input/LI, and Model Reconstruction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # 1. Ground Truth
    im0 = axes[0].imshow(np.abs(H_true), aspect='auto', cmap='jet')
    axes[0].set_title("Ground Truth |H_true|", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("OFDM Symbol")
    axes[0].set_ylabel("Subcarrier")
    plt.colorbar(im0, ax=axes[0])

    # 2. Input / LI Benchmark
    if H_in.ndim == 2 and H_in.shape == (132, 14):
        im1 = axes[1].imshow(np.abs(H_in), aspect='auto', cmap='jet')
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
    axes[2].set_title(f"DnCNN Output |H_pred|\n(NMSE: {nmse_val:.2f} dB, SSIM: {ssim_val:.4f})", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("OFDM Symbol")
    axes[2].set_ylabel("Subcarrier")
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(title_str, fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    out_pdf = os.path.join(save_dir, out_filename)
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"[Save] Exported channel reconstruction -> {out_pdf}")


# ============================================================================
# 6. MAIN TRAINING & EVALUATION ROUTINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CORAL DnCNN Domain Adaptation Channel Estimator")
    parser.add_argument('--source-dir', type=str, default=SOURCE_DIR, help='Path or folder name for source domain data')
    parser.add_argument('--target-dir', type=str, default=TARGET_DIR, help='Path or folder name for target domain data')
    parser.add_argument('--data-root', type=str, default='', help='Alias for --source-dir (single dataset mode fallback)')
    parser.add_argument('--save-dir', type=str, default=DEFAULT_SAVE_DIR, help='Directory to save outputs')
    parser.add_argument('--snr', type=int, default=DEFAULT_SNR, help='Channel SNR in dB')
    parser.add_argument('--input-type', type=str, default=DEFAULT_INPUT_TYPE, choices=['li', 'prac', 'ls'], help='Input estimation type')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate')
    parser.add_argument('--domain-weight', type=float, default=DEFAULT_DOMAIN_WEIGHT, help='CORAL loss weight (lambda)')
    parser.add_argument('--coral-layers', nargs='+', default=DEFAULT_CORAL_LAYERS, help='DnCNN intermediate blocks to align with CORAL')
    parser.add_argument('--proj-dim', type=int, default=DEFAULT_PROJ_DIM, help='Projected subspace embedding dimension')
    parser.add_argument('--only-source', action='store_true', help='Train on source domain only (no CORAL adaptation)')
    parser.add_argument('--save-features', action='store_true', help='Save intermediate activations at begin, mid, and last epochs')
    parser.add_argument('--train-frac', type=float, default=DEFAULT_TRAIN_FRAC, help='Train split fraction')
    parser.add_argument('--val-frac', type=float, default=DEFAULT_VAL_FRAC, help='Validation split fraction')
    parser.add_argument('--n-blocks', type=int, default=DEFAULT_N_BLOCKS, help='Number of residual blocks in CNNGenerator')
    parser.add_argument('--clip-extrap', action='store_true', default=DEFAULT_CLIP_EXTRAP, help='Clip extrapolation boundary errors')
    parser.add_argument('--ssim-weight-start', type=float, default=DEFAULT_SSIM_START, help='Initial SSIM loss weight')
    parser.add_argument('--ssim-weight-end', type=float, default=DEFAULT_SSIM_END, help='Final SSIM loss weight')
    parser.add_argument('--standardize', action='store_true', help='Use sample-wise standardization instead of minmax')
    parser.add_argument('--test-code', action='store_true', help='Quick smoke test mode (5 epochs on small subset)')

    args = parser.parse_args()

    # Fallback alias support
    if args.data_root:
        args.source_dir = args.data_root
        args.target_dir = args.data_root

    # Parse and validate selected layers
    selected_layers = []
    for item in args.coral_layers:
        for sub in str(item).replace(',', ' ').split():
            sub_clean = sub.strip().lower()
            if sub_clean and sub_clean not in selected_layers:
                selected_layers.append(sub_clean)

    if args.test_code:
        args.epochs = 5
        print("[test-code] Overriding epochs to 5 for quick sanity test.")

    domain_weight = 0.0 if args.only_source else args.domain_weight

    # Resolve dataset paths dynamically
    src_mat_path = get_mat_file(args.source_dir, args.snr)
    tgt_mat_path = get_mat_file(args.target_dir, args.snr)

    print("=" * 80)
    print(f"DnCNN (CNNGenerator) | Mode: {'Source-Only' if args.only_source else 'Projection-Head CORAL UDA'}")
    if not args.only_source:
        print(f"CORAL Extracted Layers: {selected_layers} (with Dedicated Projection Heads)")
        print(f"Projection Subspace Dimension: {args.proj_dim}")
        print(f"CORAL Loss Weight (lambda): {domain_weight}")
    print(f"Source Dataset: {src_mat_path}")
    print(f"Target Dataset: {tgt_mat_path}")
    print(f"SNR: {args.snr} dB | Input Type: {args.input_type.upper()} | Normalization: {'Standardize' if args.standardize else '[-1, 1] MinMax'}")
    print("=" * 80)

    # Output directory setup
    if args.save_dir and args.save_dir.strip():
        output_dir = os.path.abspath(args.save_dir)
    else:
        output_dir = os.path.join(current_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Experiment results will be saved to: {output_dir}")

    # Load Source and Target datasets
    src_dict = load_dataset_dncnn(src_mat_path, args.input_type)
    tgt_dict = load_dataset_dncnn(tgt_mat_path, args.input_type)

    H_perf_src = src_dict['H_perfect']
    H_perf_ori_src = src_dict['H_perfect_ori']
    H_in_src = src_dict['H_input']

    H_perf_tgt = tgt_dict['H_perfect']
    H_perf_ori_tgt = tgt_dict['H_perfect_ori']
    H_in_tgt = tgt_dict['H_input']

    pilot_bounds_src = (int(np.min(src_dict['pilot_rows'])), int(np.max(src_dict['pilot_rows'])) + 1,
                        int(np.min(src_dict['pilot_cols'])), int(np.max(src_dict['pilot_cols'])) + 1)
    pilot_bounds_tgt = (int(np.min(tgt_dict['pilot_rows'])), int(np.max(tgt_dict['pilot_rows'])) + 1,
                        int(np.min(tgt_dict['pilot_cols'])), int(np.max(tgt_dict['pilot_cols'])) + 1)

    idx_train_src, idx_val_src, idx_test_src = split_indices(len(H_perf_src), args.train_frac, args.val_frac)
    idx_train_tgt, idx_val_tgt, idx_test_tgt = split_indices(len(H_perf_tgt), args.train_frac, args.val_frac)

    if args.test_code:
        idx_train_src = idx_train_src[:48]; idx_val_src = idx_val_src[:16]; idx_test_src = idx_test_src[:16]
        idx_train_tgt = idx_train_tgt[:48]; idx_val_tgt = idx_val_tgt[:16]; idx_test_tgt = idx_test_tgt[:16]

    print(f"Source split -> Train: {len(idx_train_src)} | Val: {len(idx_val_src)} | Test: {len(idx_test_src)}")
    print(f"Target split -> Train: {len(idx_train_tgt)} | Val: {len(idx_val_tgt)} | Test: {len(idx_test_tgt)}")

    # Instantiate DnCNN Model with extraction layers
    model = CNNGenerator(n_blocks=args.n_blocks, extract_layers=selected_layers)
    
    # Instantiate Dedicated Projection Heads
    proj_heads = build_projection_heads(selected_layers, n_blocks=args.n_blocks, custom_proj_dim=args.proj_dim)
    print(f"\n[Model Initialized] DnCNN ({args.n_blocks} Blocks) + {len(proj_heads)} Dedicated Projection Head(s):")
    for lyr, head in proj_heads.items():
        print(f"  --> {lyr.upper()} Projection Head: [{head.in_dim} -> {head.hidden_dim} -> {head.proj_dim}]")

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.5, beta_2=0.9)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Pre-build model & projection head variables with dummy forward pass
    dummy_x = tf.zeros((args.batch_size, 132, 14, 2), dtype=tf.float32)
    _, dummy_feats = model(dummy_x, training=False, return_features=True)
    for lyr, f_t in zip(selected_layers, dummy_feats):
        proj_heads[lyr](f_t, training=False)

    # Comprehensive metric tracking across epochs
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
    mid_epoch = args.epochs // 2
    feature_checkpoint_epochs = {0: 'begin', mid_epoch: 'mid', args.epochs - 1: 'last'}

    # =========================================================================
    # COMPILED GPU TRAINING STEPS
    # =========================================================================
    @tf.function
    def _train_step_coral_phead(x_src, y_src, x_tgt, ssim_w, domain_w, standardize):
        with tf.GradientTape() as tape:
            # 1. Source forward pass
            res_src, feats_src = model(x_src, training=True, return_features=True)
            x_pred_src = x_src + res_src

            # 2. Target forward pass
            _, feats_tgt = model(x_tgt, training=True, return_features=True)

            # 3. MSE and SSIM Estimation Loss on source
            mse_val = loss_fn(y_src, x_pred_src)
            if standardize:
                max_val = tf.maximum(tf.reduce_max(y_src) - tf.reduce_min(y_src), 1e-8)
            else:
                max_val = tf.cast(2.0 if DEFAULT_LOWER_RANGE == -1 else 1.0, tf.float32)
            ssim_val = tf.image.ssim(y_src, x_pred_src, max_val=max_val)
            ssim_loss = tf.reduce_mean(1.0 - ssim_val)

            est_loss = (1.0 - ssim_w) * mse_val + ssim_w * ssim_loss
            reg_loss = 0.001 * tf.reduce_mean(tf.square(res_src))

            # 4. Multi-Head Projected CORAL Loss
            coral_losses = []
            for lyr, z_s, z_t in zip(selected_layers, feats_src, feats_tgt):
                phead = proj_heads[lyr]
                p_s = phead(z_s, training=True)
                p_t = phead(z_t, training=True)
                l_c = coral_loss_single_layer(p_s, p_t)
                coral_losses.append(l_c)

            coral_loss = tf.add_n(coral_losses) / tf.cast(len(coral_losses), tf.float32) if coral_losses else tf.constant(0.0)
            total_loss = est_loss + reg_loss + domain_w * coral_loss

        # Collect trainable variables from BOTH main model AND active projection heads
        trainable_vars = list(model.trainable_variables)
        for lyr in selected_layers:
            trainable_vars.extend(proj_heads[lyr].trainable_variables)

        grads = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        return total_loss, est_loss, coral_loss

    @tf.function
    def _train_step_source_only(x_src, y_src, ssim_w, standardize):
        with tf.GradientTape() as tape:
            res_src = model(x_src, training=True)
            x_pred_src = x_src + res_src

            mse_val = loss_fn(y_src, x_pred_src)
            if standardize:
                max_val = tf.maximum(tf.reduce_max(y_src) - tf.reduce_min(y_src), 1e-8)
            else:
                max_val = tf.cast(2.0 if DEFAULT_LOWER_RANGE == -1 else 1.0, tf.float32)
            ssim_val = tf.image.ssim(y_src, x_pred_src, max_val=max_val)
            ssim_loss = tf.reduce_mean(1.0 - ssim_val)

            est_loss = (1.0 - ssim_w) * mse_val + ssim_w * ssim_loss
            reg_loss = 0.001 * tf.reduce_mean(tf.square(res_src))
            total_loss = est_loss + reg_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, est_loss, tf.constant(0.0, dtype=tf.float32)

    print(f"\n[Train] Starting GPU-Accelerated DnCNN Projection-Head CORAL Training for {args.epochs} Epochs ...")
    t_start = time.perf_counter()
    n_batches = min(len(idx_train_src), len(idx_train_tgt)) // args.batch_size
    standardize_tf = tf.constant(args.standardize, dtype=tf.bool)
    domain_w_tf = tf.constant(domain_weight, dtype=tf.float32)

    # Subsets for quick evaluation during epochs
    eval_n_tr = min(len(idx_train_src), 64)
    eval_sub_src_tr = idx_train_src[:eval_n_tr]
    eval_sub_tgt_tr = idx_train_tgt[:eval_n_tr]

    for epoch in range(args.epochs):
        # Feature checkpointing at begin, mid, last
        if args.save_features and epoch in feature_checkpoint_epochs:
            stage = feature_checkpoint_epochs[epoch]
            src_raw, src_proj = extract_features_and_projections(model, proj_heads, H_perf_src[idx_train_src], H_in_src[idx_train_src], args.batch_size,
                                                                 selected_layers, DEFAULT_LOWER_RANGE, args.clip_extrap, pilot_bounds_src, args.standardize)
            tgt_raw, tgt_proj = extract_features_and_projections(model, proj_heads, H_perf_tgt[idx_train_tgt], H_in_tgt[idx_train_tgt], args.batch_size,
                                                                 selected_layers, DEFAULT_LOWER_RANGE, args.clip_extrap, pilot_bounds_tgt, args.standardize)
            for k, v in src_raw.items():
                saved_features[f"features_{stage}_{k}_raw_src"] = v
            for k, v in src_proj.items():
                saved_features[f"features_{stage}_{k}_proj_src"] = v
            for k, v in tgt_raw.items():
                saved_features[f"features_{stage}_{k}_raw_tgt"] = v
            for k, v in tgt_proj.items():
                saved_features[f"features_{stage}_{k}_proj_tgt"] = v
            print(f"  [Features Saved] Captured intermediate (raw) & projected activations at {stage} epoch ({epoch+1}) for layers: {selected_layers}")

        # Linear decaying SSIM weight schedule
        if args.epochs > 1:
            epoch_ssim_w = args.ssim_weight_start + (epoch / (args.epochs - 1)) * (args.ssim_weight_end - args.ssim_weight_start)
        else:
            epoch_ssim_w = args.ssim_weight_start
        ssim_w_tf = tf.constant(epoch_ssim_w, dtype=tf.float32)

        p_src = np.random.permutation(len(idx_train_src))
        p_tgt = np.random.permutation(len(idx_train_tgt))
        train_in_src = H_in_src[idx_train_src][p_src]
        train_perf_src = H_perf_src[idx_train_src][p_src]
        train_in_tgt = H_in_tgt[idx_train_tgt][p_tgt]
        train_perf_tgt = H_perf_tgt[idx_train_tgt][p_tgt]

        ep_total_l, ep_est_l, ep_coral_l = 0.0, 0.0, 0.0

        for b in range(n_batches):
            sl = slice(b * args.batch_size, (b + 1) * args.batch_size)
            x_s, y_s, _, _ = preprocess_batch(train_perf_src[sl], train_in_src[sl], DEFAULT_LOWER_RANGE,
                                              args.clip_extrap, pilot_bounds_src, args.standardize)
            x_t, _, _, _ = preprocess_batch(train_perf_tgt[sl], train_in_tgt[sl], DEFAULT_LOWER_RANGE,
                                            args.clip_extrap, pilot_bounds_tgt, args.standardize)

            if args.only_source:
                tot_l, est_l, cor_l = _train_step_source_only(x_s, y_s, ssim_w_tf, standardize_tf)
            else:
                tot_l, est_l, cor_l = _train_step_coral_phead(x_s, y_s, x_t, ssim_w_tf, domain_w_tf, standardize_tf)

            ep_total_l += tot_l.numpy()
            ep_est_l += est_l.numpy()
            ep_coral_l += cor_l.numpy()

        history['train_loss'].append(ep_total_l / max(n_batches, 1))
        history['train_est_loss'].append(ep_est_l / max(n_batches, 1))
        history['train_coral_loss'].append(ep_coral_l / max(n_batches, 1))

        # Track metrics (NMSE, MSE, SSIM) on train & val splits every epoch or periodic
        eval_interval = 1 if (args.epochs <= 50 or args.test_code) else 5
        if (epoch + 1) % eval_interval == 0 or epoch == args.epochs - 1:
            # 1. Source Train & Val
            pred_src_tr = infer_channel(model, H_perf_src[eval_sub_src_tr], H_in_src[eval_sub_src_tr], args.batch_size,
                                        DEFAULT_LOWER_RANGE, args.clip_extrap, pilot_bounds_src, args.standardize)
            pred_src_val = infer_channel(model, H_perf_src[idx_val_src], H_in_src[idx_val_src], args.batch_size,
                                         DEFAULT_LOWER_RANGE, args.clip_extrap, pilot_bounds_src, args.standardize)
            
            # 2. Target Train & Val
            pred_tgt_tr = infer_channel(model, H_perf_tgt[eval_sub_tgt_tr], H_in_tgt[eval_sub_tgt_tr], args.batch_size,
                                        DEFAULT_LOWER_RANGE, args.clip_extrap, pilot_bounds_tgt, args.standardize)
            pred_tgt_val = infer_channel(model, H_perf_tgt[idx_val_tgt], H_in_tgt[idx_val_tgt], args.batch_size,
                                         DEFAULT_LOWER_RANGE, args.clip_extrap, pilot_bounds_tgt, args.standardize)

            # Compute metrics
            nmse_s_tr = 10.0 * np.log10(compute_nmse(pred_src_tr, H_perf_src[eval_sub_src_tr]) + 1e-30)
            nmse_s_val = 10.0 * np.log10(compute_nmse(pred_src_val, H_perf_src[idx_val_src]) + 1e-30)
            nmse_t_tr = 10.0 * np.log10(compute_nmse(pred_tgt_tr, H_perf_tgt[eval_sub_tgt_tr]) + 1e-30)
            nmse_t_val = 10.0 * np.log10(compute_nmse(pred_tgt_val, H_perf_tgt[idx_val_tgt]) + 1e-30)

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

            print(f"Epoch {epoch+1:03d}/{args.epochs:03d} | Total Loss: {ep_total_l/n_batches:.4f} "
                  f"(Est: {ep_est_l/n_batches:.4f}, Proj-CORAL: {ep_coral_l/n_batches:.4f}) | "
                  f"Target NMSE: {nmse_t_val:.2f} dB (Src: {nmse_s_val:.2f} dB) | Target SSIM: {ssim_t_val:.4f}")

    total_time = time.perf_counter() - t_start
    print(f"\n[Done] Training completed in {total_time:.2f} seconds ({total_time/args.epochs:.3f} s/epoch).")

    # =========================================================================
    # FINAL FULL DATASET INFERENCE & EVALUATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("                      FINAL TEST PERFORMANCE SUMMARY                  ")
    print("=" * 80)
    test_pred_src = infer_channel(model, H_perf_src[idx_test_src], H_in_src[idx_test_src], args.batch_size,
                                  DEFAULT_LOWER_RANGE, args.clip_extrap, pilot_bounds_src, args.standardize)
    test_pred_tgt = infer_channel(model, H_perf_tgt[idx_test_tgt], H_in_tgt[idx_test_tgt], args.batch_size,
                                  DEFAULT_LOWER_RANGE, args.clip_extrap, pilot_bounds_tgt, args.standardize)

    train_pred_src_sample = infer_channel(model, H_perf_src[idx_train_src[:10]], H_in_src[idx_train_src[:10]], args.batch_size,
                                          DEFAULT_LOWER_RANGE, args.clip_extrap, pilot_bounds_src, args.standardize)
    train_pred_tgt_sample = infer_channel(model, H_perf_tgt[idx_train_tgt[:10]], H_in_tgt[idx_train_tgt][:10], args.batch_size,
                                          DEFAULT_LOWER_RANGE, args.clip_extrap, pilot_bounds_tgt, args.standardize)

    nmse_src_db = 10.0 * np.log10(compute_nmse(test_pred_src, H_perf_src[idx_test_src]) + 1e-30)
    nmse_tgt_db = 10.0 * np.log10(compute_nmse(test_pred_tgt, H_perf_tgt[idx_test_tgt]) + 1e-30)
    mmse_src = compute_mmse(test_pred_src, H_perf_src[idx_test_src])
    mmse_tgt = compute_mmse(test_pred_tgt, H_perf_tgt[idx_test_tgt])
    ssim_src = compute_ssim_batch(test_pred_src, H_perf_src[idx_test_src])
    ssim_tgt = compute_ssim_batch(test_pred_tgt, H_perf_tgt[idx_test_tgt])

    print(f"  Source Domain (Test Set) -> NMSE: {nmse_src_db:.2f} dB | MMSE: {mmse_src:.6e} | SSIM: {ssim_src:.4f}")
    print(f"  Target Domain (Test Set) -> NMSE: {nmse_tgt_db:.2f} dB | MMSE: {mmse_tgt:.6e} | SSIM: {ssim_tgt:.4f}")
    print("=" * 80)

    # 1. Save Exported Test MAT Files (for MATLAB benchmarking)
    save_test_channel_mat(os.path.join(output_dir, 'testChannel_source.mat'), H_perf_src[idx_test_src],
                          H_perf_ori_src[idx_test_src], H_in_src[idx_test_src], test_pred_src, src_dict,
                          idx_test_src, args.snr, f"DnCNN_pHead_{args.input_type.upper()}")

    save_test_channel_mat(os.path.join(output_dir, 'testChannel_target.mat'), H_perf_tgt[idx_test_tgt],
                          H_perf_ori_tgt[idx_test_tgt], H_in_tgt[idx_test_tgt], test_pred_tgt, tgt_dict,
                          idx_test_tgt, args.snr, f"DnCNN_pHead_{args.input_type.upper()}")

    # 2. Save Plotted Channel Samples to sample_reconstructions.mat
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

        'pilot_rows': src_dict['pilot_rows'] + 1,
        'pilot_cols': src_dict['pilot_cols'] + 1,
        'snr': args.snr,
        'input_type': args.input_type,
        'model_type': 'DnCNN_pHead'
    }
    savemat(os.path.join(output_dir, 'sample_reconstructions.mat'), samples_dict)
    print(f"[Save] Exported sample reconstruction grids MAT file -> {os.path.join(output_dir, 'sample_reconstructions.mat')}")

    # 3. Save Evaluation Results
    eval_dict = {
        'nmse_test_src_db': nmse_src_db, 'mmse_test_src': mmse_src, 'ssim_test_src': ssim_src,
        'nmse_test_tgt_db': nmse_tgt_db, 'mmse_test_tgt': mmse_tgt, 'ssim_test_tgt': ssim_tgt,
        'nmse_test_db': nmse_tgt_db, 'mmse_test': mmse_tgt, 'ssim_test': ssim_tgt,
        'snr': args.snr, 'input_type': args.input_type, 'proj_dim': args.proj_dim,
        'coral_layers': np.array(selected_layers)
    }
    savemat(os.path.join(output_dir, 'evaluation_results.mat'), eval_dict)

    # 4. Save Comprehensive Training History MAT
    savemat(os.path.join(output_dir, 'training_history.mat'), {k: np.array(v) for k, v in history.items()})
    print(f"[Save] Exported training history MAT -> {os.path.join(output_dir, 'training_history.mat')}")

    # 5. Save Extracted Features MAT if requested
    if args.save_features and saved_features:
        saved_features['selected_layers'] = np.array(selected_layers)
        saved_features['train_indices_src'] = idx_train_src
        saved_features['train_indices_tgt'] = idx_train_tgt
        savemat(os.path.join(output_dir, 'extracted_features.mat'), saved_features, do_compression=True)
        print(f"[Save] Exported extracted raw and projected features -> {os.path.join(output_dir, 'extracted_features.mat')}")

    # 6. Render All PDF Visualizations
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

        # F. Individual Channel Reconstructions (Ground Truth, Input/LI, Output)
        save_single_reconstruction_pdf(samples_dict['source_train_true'], samples_dict['source_train_in'], samples_dict['source_train_pred'],
                                       "Source Domain - Training Sample", "recon_source_train.pdf", output_dir)
        save_single_reconstruction_pdf(samples_dict['source_test_true'], samples_dict['source_test_in'], samples_dict['source_test_pred'],
                                       "Source Domain - Testing Sample", "recon_source_test.pdf", output_dir)
        save_single_reconstruction_pdf(samples_dict['target_train_true'], samples_dict['target_train_in'], samples_dict['target_train_pred'],
                                       "Target Domain - Training Sample", "recon_target_train.pdf", output_dir)
        save_single_reconstruction_pdf(samples_dict['target_test_true'], samples_dict['target_test_in'], samples_dict['target_test_pred'],
                                       "Target Domain - Testing Sample", "recon_target_test.pdf", output_dir)
    except Exception as e:
        print(f"[Plot Warning] Failed to render some PDF plots: {e}")

    print(f"\n[Done] Projection-Head CORAL DnCNN execution completed successfully. Output saved to -> {output_dir}")


if __name__ == '__main__':
    main()
