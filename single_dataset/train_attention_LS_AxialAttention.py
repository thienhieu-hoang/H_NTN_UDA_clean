"""
Single-Dataset Attention-based Channel Estimator with Dynamic MSE & SSIM Loss Schedule (OpenNTN)
=====================================================================================
Train, evaluate, and test a Attention-based Generator on ONE SNR split of the OpenNTN dataset.
All data (train / val / test) come from the *same* SNR folder.
This script uses a dynamic scheduling approach for combining MSE and SSIM:
  - Start with a large SSIM weight to help the model learn coarse layout structures first.
  - Linearly decay the SSIM weight to a smaller value over the training epochs.
  - Let the MSE loss dominate in the later epochs to fine-tune the absolute coefficient values.
====================================================================================
Workflow: 
- Input: 
    - H_LS sequence (only values at the pilot positions)
- Process: 
    - Apply min-max scaling (min-max of the input) to all input and corresponding label H_true (same scaling for all) to scale the values to [-1, 1]. 
    ----------------------------------------------------
    Go through Attention + FFN blocks:
        - Attention block computes relationships between the pilot values.
        - Fully connected layer upsamples/expands the features to fill the unknown positions.
        - Convolutional layers refine the reconstructed 132 x 14 grid.
    ----------------------------------------------------
    - De-scale the output with the min-max of the input.
- Output: 
    - H_hat_LS (132 x 14 grid) 
====================================================================================
Usage
-----
    python train_attention_LS.py --snr 10 --loss-type combined --ssim-weight-start 0.95 --ssim-weight-end 0.05 --save-model

    # Example for running a quick smoke test
    python train_attention_LS.py --snr 10 --test-code --save-model
"""

# ── Standard library ────────────────────────────────────────────────────────
import os
import sys
import time
import argparse

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
# NumPy 2.0 compatibility monkey-patch for older TensorFlow/Keras releases
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

# Register string representations in sctypeDict for np.dtype('string_')
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

import scipy.io
import tensorflow as tf
from tensorflow.image import ssim as tf_ssim

# ============================================================================
# DEFAULT HYPER-PARAMETERS
# ============================================================================
DEFAULT_SNR         = 10           # dB
DEFAULT_INPUT_TYPE  = 'ls'         # 'ls' or 'ls_ori' (sparse pilots)
DEFAULT_EPOCHS      = 200
DEFAULT_BATCH_SIZE  = 16
DEFAULT_LR          = 1e-4
DEFAULT_TRAIN_FRAC  = 0.70
DEFAULT_VAL_FRAC    = 0.15
DEFAULT_LOWER_RANGE = -1           # minmax scaling range: -1 -> [-1, 1]
DEFAULT_SSIM_START  = 0.95         # Initial SSIM weight
DEFAULT_SSIM_END    = 0.05         # Final SSIM weight
TEST_CODE_N_TRAIN   = 48
TEST_CODE_N_VAL     = 16
TEST_CODE_N_TEST    = 16
TEST_CODE_EPOCHS    = 5
DEFAULT_SAVE_MODEL  = False
DEFAULT_SAVE_DIR    = ''
DEFAULT_DATA_ROOT   = ''

# ============================================================================
# Path setup
# ============================================================================
def _setup_paths():
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        this_dir = os.getcwd()

    project_root = os.path.abspath(os.path.join(this_dir, '..'))
    for p in [project_root,
              os.path.join(project_root, 'Domain_Adversarial', 'helper'),
              os.path.join(project_root, 'JMMD', 'helper')]:
        if p not in sys.path:
            sys.path.insert(0, p)
    return this_dir, project_root

THIS_DIR, PROJECT_ROOT = _setup_paths()

# Import standardization helpers
from utils import standardizeScaler_ha02, deStandardize_ha02
from utils_GAN import SameShapeBlock, AxialAttention2D, reflect_padding_2d

# =============================================================================
# 1. TRANSFORMER ENCODER BLOCK (Attention Pre-processor)
# =============================================================================
class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_pilot_elems=88, num_channels=2, num_heads=2, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.num_pilot_elems = num_pilot_elems
        self.num_channels = num_channels
        self.num_heads = num_heads
        
        self.in_dim = num_pilot_elems * num_channels
        self.head_dim = self.in_dim // num_heads
        
        self.fc1 = tf.keras.layers.Dense(3 * self.in_dim, name="qkv_projection")
        self.fc2 = tf.keras.layers.Dense(self.in_dim, name="attn_out_projection")
        
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm_1")
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm_2")
        
        self.ffn_dense1 = tf.keras.layers.Dense(self.in_dim * 2, name="ffn_dense1")
        self.ffn_dense2 = tf.keras.layers.Dense(self.in_dim, name="ffn_dense2")

    def call(self, inputs):
        B = tf.shape(inputs)[0]
        x_flat = tf.reshape(inputs, [B, self.in_dim])
        
        qkv = self.fc1(x_flat)
        qkv = tf.reshape(qkv, [B, 3, self.num_heads, self.head_dim])
        Q = qkv[:, 0, :, :]
        K = qkv[:, 1, :, :]
        V = qkv[:, 2, :, :]
        
        scale = tf.cast(tf.sqrt(self.num_pilot_elems / self.num_heads), dtype=tf.float32)
        
        Q_exp = tf.expand_dims(Q, axis=-1)
        K_exp = tf.expand_dims(K, axis=-2)
        
        scores = tf.matmul(Q_exp, K_exp) / scale
        attn_weights = tf.nn.softmax(scores, axis=-1)
        
        V_exp = tf.expand_dims(V, axis=-1)
        attn_out = tf.squeeze(tf.matmul(attn_weights, V_exp), axis=-1)
        
        attn_out_flat = tf.reshape(attn_out, [B, self.in_dim])
        attn_proj = self.fc2(attn_out_flat)
        
        x_norm1 = self.ln1(x_flat + attn_proj)
        
        ffn1 = tf.nn.gelu(self.ffn_dense1(x_norm1))
        ffn_out = self.ffn_dense2(ffn1)
        out = self.ln2(x_norm1 + ffn_out)
        
        return tf.reshape(out, [B, self.num_pilot_elems, self.num_channels])

# =============================================================================
# 2. RESIDUAL CONVOLUTIONAL DECODER BLOCK (Decoder + Upsampler)
# =============================================================================
class ResidualConvDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_pilot_elems=88, total_grid_elems=1848, n_filter=32, **kwargs):
        super(ResidualConvDecoderBlock, self).__init__(**kwargs)
        self.num_pilot_elems = num_pilot_elems
        self.total_grid_elems = total_grid_elems
        self.num_subcarriers = total_grid_elems // 14
        self.n_filter = n_filter
        
        self.conv1 = tf.keras.layers.Conv2D(filters=n_filter, kernel_size=(2, 2), padding='same', name="conv1")
        self.res_conv1 = tf.keras.layers.Conv2D(filters=n_filter, kernel_size=(2, 2), padding='same', name="res_conv1")
        self.relu = tf.keras.layers.ReLU()
        self.res_conv2 = tf.keras.layers.Conv2D(filters=n_filter, kernel_size=(2, 2), padding='same', name="res_conv2")
        self.norm = tf.keras.layers.BatchNormalization(name="batch_norm")
        self.fc_upsample = tf.keras.layers.Dense(total_grid_elems, name="fc_upsample")
        
        C_hidden = 2 * n_filter
        self.axial_attention = AxialAttention2D(channels=C_hidden, name="axial_attention")
        self.post_block1 = SameShapeBlock(filters=C_hidden, name="post_block1")
        self.post_block2 = SameShapeBlock(filters=C_hidden // 2, name="post_block2")
        self.conv_out = tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3), padding='valid', name="conv_out")

    def call(self, inputs, training=False):
        B = tf.shape(inputs)[0]
        x_img = tf.expand_dims(inputs, axis=-1)
        
        h1 = self.conv1(x_img)
        res = self.res_conv1(h1)
        res = self.relu(res)
        res = self.res_conv2(res)
        h2 = self.norm(h1 + res, training=training)
        
        h2_trans = tf.transpose(h2, [0, 3, 2, 1])
        h2_upsampled = self.fc_upsample(h2_trans)
        h2_upsampled = tf.transpose(h2_upsampled, [0, 3, 2, 1])
        
        h2_grid = tf.reshape(h2_upsampled, [B, self.num_subcarriers, 14, 2, self.n_filter])
        h2_features = tf.reshape(h2_grid, [B, self.num_subcarriers, 14, 2 * self.n_filter])
        
        out_attn = self.axial_attention(h2_features)
        out = self.post_block1(out_attn, training=training)
        out = self.post_block2(out, training=training)
        
        out = reflect_padding_2d(out, pad_h=1, pad_w=1)
        out_grid = self.conv_out(out)
        return out_grid

# =============================================================================
# 3. COMPLETE HA02 MODEL (Keras Model)
# =============================================================================
class HA02Model(tf.keras.Model):
    def __init__(self, num_pilot_elems=88, total_grid_elems=1848, num_channels=2, num_heads=2, n_filter=32, **kwargs):
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
        encoder_out = self.encoder(inputs)
        full_grid_out = self.decoder(encoder_out, training=training)
        return full_grid_out

# =============================================================================
# 4. HUBER LOSS FUNCTION
# =============================================================================
class HuberLoss(tf.keras.losses.Loss):
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
# Helper Utilities
# =============================================================================
DATA_FOLDER_NAME = 'DUR200ns_27G_600km_r15km_20to30mps'
SNR_FOLDER_MAP = {
    -10: '-10dB',
    -5:  '-5dB',
    0:   '0dB',
    5:   '5dB',
    10:  '10dB',
    15:  '15dB',
}

def find_channel_mat_file(base_dir: str) -> str:
    """
    Find and return the primary channel .mat dataset file in base_dir.
    Prioritizes known dataset filenames (e.g. 'matlabNTN.mat', 'channel_dur_randomizedUE.mat')
    and filters out output/artifact/intermediate files in case other .mat files exist.
    """
    if not os.path.exists(base_dir):
        return None

    # Preferred dataset filename candidates (checked in order of priority)
    primary_dataset_names = [
        'matlabNTN.mat',
        'channel_dur_randomizedUE.mat',
        'channel_dur.mat',
        'channel_sur.mat',
        'channel.mat',
        'dataset.mat',
        'data.mat',
    ]

    # 1. Direct match with prioritized filenames in base_dir
    for name in primary_dataset_names:
        candidate = os.path.join(base_dir, name)
        if os.path.isfile(candidate):
            return candidate

    # 2. Case-insensitive search for primary dataset filenames
    if os.path.isdir(base_dir):
        files_in_dir = os.listdir(base_dir)
        for name in primary_dataset_names:
            for f in files_in_dir:
                if f.lower() == name.lower() and os.path.isfile(os.path.join(base_dir, f)):
                    return os.path.join(base_dir, f)

    # 3. Exclude any known output / artifact / temporary .mat files
    excluded_prefixes = (
        'testChannel', 'inferredChannel', 'training_history',
        'channel_grids', 'synthesized', 'evaluation_results',
        'sample_', 'extracted_features', 'loss_', 'nmse_', 'best_', 'final_'
    )

    # Search for any valid channel .mat file inside base_dir
    for root, _, files in os.walk(base_dir):
        for f in sorted(files):
            if f.endswith('.mat') and not f.startswith(excluded_prefixes):
                return os.path.join(root, f)

    return None


def get_data_path(data_root: str, snr: int, is_test_code: bool = False) -> str:
    """
    Robustly locate the .mat data file for the requested SNR, supporting:
    - SNR folder variations: 'SNR_-10dB', '-10dB', 'SNR_-10', '-10', 'SNR_-10dB'
    - Dataset filename variations: 'matlabNTN.mat', 'channel_dur_randomizedUE.mat', etc.
    - Relative paths from workspace root or script directory
    - Scenario name substring matching in generatedChan/MATLAB and generatedChan/OpenNTN
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    candidate_roots = []
    if data_root:
        if os.path.isabs(data_root):
            candidate_roots.append(data_root)
        else:
            candidate_roots.append(os.path.abspath(data_root))
            candidate_roots.append(os.path.join(project_root, data_root))
            candidate_roots.append(os.path.join(script_dir, data_root))
            
            # Scenario substring search in generatedChan/
            for parent in [os.path.join(project_root, 'generatedChan', 'MATLAB'),
                           os.path.join(project_root, 'generatedChan', 'OpenNTN')]:
                if os.path.isdir(parent):
                    base_name = os.path.basename(data_root.rstrip('\\/'))
                    for entry in os.listdir(parent):
                        if base_name in entry:
                            candidate_roots.append(os.path.join(parent, entry))

    # Add default fallback directories
    candidate_roots.extend([
        os.path.join(project_root, 'generatedChan', 'OpenNTN', DATA_FOLDER_NAME),
        os.path.join(project_root, 'generatedChan', 'OpenNTN', 'DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps'),
        os.path.join(project_root, 'generatedChan', 'MATLAB', 'sampleWiseDoppler_wGeometry_A100_2p18e9_600km_70deg_30kHz'),
        os.path.join(project_root, 'generatedChan', 'MATLAB', 'A100_2p18e9_600km_70deg_30kHz')
    ])

    snr_variations = [
        f"SNR_{snr}dB",
        f"{snr}dB",
        f"SNR_{snr}",
        f"{snr}",
        f"SNR_{snr:02d}dB",
    ]

    for root in candidate_roots:
        if not os.path.isdir(root):
            continue
        # 1. Try each SNR subfolder variation
        for snr_var in snr_variations:
            snr_dir = os.path.join(root, snr_var)
            if os.path.isdir(snr_dir):
                mat_file = find_channel_mat_file(snr_dir)
                if mat_file:
                    return mat_file

        # 2. Check if the root itself is an SNR directory (e.g. user passed path directly to SNR folder)
        root_base = os.path.basename(root.rstrip('\\/'))
        if root_base in snr_variations or root_base.lower() in [v.lower() for v in snr_variations]:
            mat_file = find_channel_mat_file(root)
            if mat_file:
                return mat_file

    raise FileNotFoundError(
        f"Could not find any channel .mat data files for SNR={snr} in any searched location.\n"
        f"Searched roots: {candidate_roots}"
    )


def load_mat_data(mat_path: str, input_type: str):
    try:
        mat = scipy.io.loadmat(mat_path)
        is_hdf5 = False
    except NotImplementedError:
        import h5py
        is_hdf5 = True

    if is_hdf5:
        def h5_to_complex(val):
            if isinstance(val, np.ndarray) and val.dtype.names is not None:
                if 'real' in val.dtype.names and 'imag' in val.dtype.names:
                    return val['real'] + 1j * val['imag']
            return val

        with h5py.File(mat_path, 'r') as f:
            H_perfect = h5_to_complex(f['H_perfect'][()])
            if H_perfect.ndim == 3:
                if H_perfect.shape[1] == 14 and H_perfect.shape[2] == 132:
                    H_perfect = np.transpose(H_perfect, (0, 2, 1))

            # Load H_perfect_ori (original true channel before Doppler compensation)
            H_perfect_ori = None
            for key in ['H_perfect_ori', 'H_perfect_original', 'H_true_ori', 'H_ori']:
                if key in f:
                    H_perfect_ori = h5_to_complex(f[key][()])
                    if H_perfect_ori.ndim == 3:
                        if H_perfect_ori.shape[1] == 14 and H_perfect_ori.shape[2] == 132:
                            H_perfect_ori = np.transpose(H_perfect_ori, (0, 2, 1))
                    break
            if H_perfect_ori is None:
                H_perfect_ori = H_perfect

            # Reconstruct dictionary of other variables
            mat_dict = {}
            for k in f.keys():
                if k.startswith('#'):
                    continue
                mat_dict[k] = h5_to_complex(f[k][()])
                if k in ['pilot_rows', 'pilot_cols']:
                    mat_dict[k] = np.squeeze(mat_dict[k])
                elif isinstance(mat_dict[k], np.ndarray) and mat_dict[k].ndim == 3:
                    if mat_dict[k].shape[1] == 14 and mat_dict[k].shape[2] == 132:
                        mat_dict[k] = np.transpose(mat_dict[k], (0, 2, 1))

            input_key_map = {
                'prac': 'H_prac',
                'li': 'H_li',
                'li_ori': 'H_li_ori',
                'ls': 'H_ls_pilots',
                'ls_ori': 'H_ls_pilots_ori'
            }
            input_key = input_key_map.get(input_type, 'H_ls_pilots')
            if input_key not in mat_dict:
                for alt in [input_key, 'H_ls_pilots', 'H_ls_pilots_ori', 'H_li', 'H_li_ori', 'H_prac']:
                    if alt in mat_dict:
                        input_key = alt
                        break

            H_input_pilots = mat_dict[input_key]
            pilot_cols = mat_dict['pilot_cols'].squeeze() - 1
            pilot_rows = mat_dict['pilot_rows'].squeeze() - 1

            if H_input_pilots.ndim == 3:
                H_input_pilots = H_input_pilots[:, pilot_cols, pilot_rows]

            H_li_benchmark_grid = mat_dict.get('H_li', mat_dict.get('H_li_ori', mat_dict.get('H_perfect')))
            if H_li_benchmark_grid.ndim == 2:
                H_li_benchmark_grid = H_perfect

    else:
        H_perfect = mat['H_perfect'].T
        
        # Load H_perfect_ori
        H_perfect_ori = None
        for key in ['H_perfect_ori', 'H_perfect_original', 'H_true_ori', 'H_ori']:
            if key in mat:
                H_perfect_ori = mat[key].T
                break
        if H_perfect_ori is None:
            H_perfect_ori = H_perfect
        
        input_key_map = {
            'prac': 'H_prac',
            'li': 'H_li',
            'li_ori': 'H_li_ori',
            'ls': 'H_ls_pilots',
            'ls_ori': 'H_ls_pilots_ori'
        }
        input_key = input_key_map.get(input_type, 'H_ls_pilots')
        
        if input_key not in mat or mat[input_key].size == 0:
            for alt in [input_key, 'H_ls_pilots', 'H_ls_pilots_ori', 'H_li', 'H_li_ori', 'H_prac']:
                if alt in mat and isinstance(mat[alt], np.ndarray) and mat[alt].size > 0:
                    input_key = alt
                    break
                    
        H_input_pilots = mat[input_key].T
        pilot_cols = mat['pilot_cols'].squeeze() - 1
        pilot_rows = mat['pilot_rows'].squeeze() - 1
        
        if H_input_pilots.ndim == 3:
            H_input_pilots = H_input_pilots[:, pilot_cols, pilot_rows]
            
        H_li_benchmark_grid = mat.get('H_li', mat.get('H_li_ori', mat['H_perfect'])).T
        if H_li_benchmark_grid.ndim == 2:
            H_li_benchmark_grid = H_perfect
            
        mat_dict = mat

    return H_perfect, H_input_pilots, H_li_benchmark_grid, mat_dict, H_perfect_ori

def split_indices(N: int, train_frac: float, val_frac: float, seed: int = 1234):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)
    
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)
    
    idx_train = indices[:n_train]
    idx_val   = indices[n_train:n_train + n_val]
    idx_test  = indices[n_train + n_val:]
    return idx_train, idx_val, idx_test

def complx2real(x: np.ndarray) -> np.ndarray:
    return np.stack([x.real, x.imag], axis=-1)

def minmaxScaler_ha02(x, y, lower_range=-1):
    B = tf.shape(x)[0]
    x_min = tf.reduce_min(x, axis=1)
    x_max = tf.reduce_max(x, axis=1)
    
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

def preprocess_batch(H_perf_batch: np.ndarray, H_in_batch: np.ndarray, lower_range: int, standardize: bool = False):
    y = complx2real(H_perf_batch)
    x = complx2real(H_in_batch)
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    if standardize:
        x_sc, y_sc, x_val1, x_val2 = standardizeScaler_ha02(x, y)
    else:
        x_sc, y_sc, x_val1, x_val2 = minmaxScaler_ha02(x, y, lower_range)
    return x_sc, y_sc, x_val1, x_val2

def compute_mmse(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    return float(np.mean(np.abs(H_pred - H_true)**2))

def compute_nmse(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    diff_sq = np.mean(np.abs(H_pred - H_true)**2)
    ref_sq  = np.mean(np.abs(H_true)**2)
    return float(diff_sq / max(ref_sq, 1e-30))

def compute_nmse_db(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    val = compute_nmse(H_pred, H_true)
    return float(10.0 * np.log10(val + 1e-30))

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

def save_channel_plots_pdf(H_perf_sample: np.ndarray,
                           H_in_sample: np.ndarray,
                           H_pred_sample: np.ndarray,
                           pilot_rows: np.ndarray,
                           pilot_cols: np.ndarray,
                           input_type: str, save_dir: str,
                           prefix: str = 'test'):
    try:
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)

        # 1. Perfect Reference Channel
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(H_perf_sample.real, aspect='auto', cmap='viridis')
        fig.colorbar(im, ax=ax)
        ax.set_title(f'Perfect Reference Channel (Real Part) - {prefix.capitalize()} Sample 1', fontsize=14)
        ax.set_xlabel('Subcarrier Index' if H_perf_sample.shape[0] == 132 else 'Symbol Index', fontsize=12)
        ax.set_ylabel('Symbol Index' if H_perf_sample.shape[0] == 132 else 'Subcarrier Index', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'H_perfect_{prefix}_sample1.pdf'), format='pdf')
        plt.close(fig)

        H_in_grid = np.zeros(H_perf_sample.shape, dtype=np.complex64)
        H_in_grid[pilot_rows, pilot_cols] = H_in_sample
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(H_in_grid.real, aspect='auto', cmap='viridis')
        fig.colorbar(im, ax=ax)
        ax.set_title(f'Sparse Input Pilots H_{input_type} (Real Part) - {prefix.capitalize()} Sample 1', fontsize=14)
        ax.set_xlabel('Subcarrier Index' if H_perf_sample.shape[0] == 132 else 'Symbol Index', fontsize=12)
        ax.set_ylabel('Symbol Index' if H_perf_sample.shape[0] == 132 else 'Subcarrier Index', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'H_{input_type}_{prefix}_sample1.pdf'), format='pdf')
        plt.close(fig)

        # 3. Model Output
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(H_pred_sample.real, aspect='auto', cmap='viridis')
        fig.colorbar(im, ax=ax)
        ax.set_title(f'HA02 Model Output H_{input_type}_attention (Real Part) - {prefix.capitalize()} Sample 1', fontsize=14)
        ax.set_xlabel('Subcarrier Index' if H_perf_sample.shape[0] == 132 else 'Symbol Index', fontsize=12)
        ax.set_ylabel('Symbol Index' if H_perf_sample.shape[0] == 132 else 'Subcarrier Index', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'H_{input_type}_attention_{prefix}_sample1.pdf'), format='pdf')
        plt.close(fig)

        print(f'[PDF Export] {prefix.capitalize()} channel grid heatmaps saved to: {save_dir}')
    except Exception as e:
        print(f'[PDF Export Warning] Failed to export {prefix} channel heatmaps: {e}')

def save_loss_plot_pdf(history: dict, save_dir: str):
    try:
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(history['train_loss'], label='Train Total Loss', color='blue')
        ax.plot(history['val_loss'], label='Val Total Loss', color='red')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training History - Loss', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_total.pdf'), format='pdf')
        plt.close(fig)

        # MSE Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(history['train_mse'], label='Train MSE', color='blue')
        ax.plot(history['val_mse'], label='Val MSE', color='red')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
        ax.set_title('Training History - MSE', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_mse.pdf'), format='pdf')
        plt.close(fig)

        # SSIM Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(history['train_ssim'], label='Train SSIM Loss', color='blue')
        ax.plot(history['val_ssim'], label='Val SSIM Loss', color='red')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('1 - SSIM', fontsize=12)
        ax.set_title('Training History - SSIM Loss', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_ssim.pdf'), format='pdf')
        plt.close(fig)

        print(f'[PDF Export] Loss history plots saved to: {save_dir}')
    except Exception as e:
        print(f'[PDF Export Warning] Failed to export loss history plots: {e}')

@tf.function
def _train_step(model, x_scaled, y_scaled, optimizer, loss_fn, lower_range, ssim_weight, use_huber=False, huber_delta=1.0, standardize=False):
    x_scaled = tf.cast(x_scaled, tf.float32)
    y_scaled = tf.cast(y_scaled, tf.float32)
    with tf.GradientTape() as tape:
        y_pred = model(x_scaled, training=True)
        if use_huber:
            err = tf.abs(y_pred - y_scaled)
            huber_err = tf.where(
                err <= huber_delta,
                0.5 * tf.square(err),
                huber_delta * (err - 0.5 * huber_delta)
            )
            total_loss = tf.reduce_mean(huber_err)
            mse_loss = tf.reduce_mean(tf.square(y_pred - y_scaled))
            ssim_loss = tf.constant(0.0)
        else:
            mse_loss = loss_fn(y_scaled, y_pred)
            if standardize:
                max_val = tf.reduce_max(y_scaled) - tf.reduce_min(y_scaled)
                max_val = tf.maximum(max_val, 1e-8)
            else:
                max_val = tf.cast(2.0 if lower_range == -1 else 1.0, tf.float32)
            ssim_val = tf_ssim(y_scaled, y_pred, max_val=max_val)
            ssim_loss = tf.reduce_mean(1.0 - ssim_val)
            total_loss = (1.0 - ssim_weight) * mse_loss + ssim_weight * ssim_loss
            
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, mse_loss, ssim_loss

def infer_channel(model, H_perfect_data, H_input_pilots, batch_size=16, lower_range=-1, standardize=False):
    N_samples = H_perfect_data.shape[0]
    H_pred_all = []
    
    for i in range(0, N_samples, batch_size):
        batch_idx = range(i, min(i + batch_size, N_samples))
        h_p = H_perfect_data[batch_idx]
        h_i = H_input_pilots[batch_idx]
        
        x_sc, y_sc, x_val1, x_val2 = preprocess_batch(h_p, h_i, lower_range, standardize=standardize)
        y_pred_sc = model(x_sc, training=False)
        if standardize:
            y_pred = deStandardize_ha02(y_pred_sc, x_val1, x_val2)
        else:
            y_pred = deMinMax_ha02(y_pred_sc, x_val1, x_val2, lower_range)
        
        y_pred_np = y_pred.numpy()
        H_pred_complex = y_pred_np[..., 0] + 1j * y_pred_np[..., 1]
        H_pred_all.append(H_pred_complex)
        
    return np.concatenate(H_pred_all, axis=0)

def export_model_to_onnx(model: tf.keras.Model, save_path: str, input_shape):
    """Export Keras model to ONNX."""
    try:
        import tf2onnx
    except ImportError:
        print('[ONNX] Installing tf2onnx package for ONNX model export ...')
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tf2onnx'])
        import tf2onnx

    try:
        spec = (tf.TensorSpec(input_shape, tf.float32, name='input_channel'),)
        model_proto, _ = tf2onnx.convert.from_keras(
            model, input_signature=spec, output_path=save_path)
        print(f'[ONNX Export] Saved ONNX model (architecture + weights) -> {save_path}')
    except Exception as e:
        print(f'[ONNX Export Warning] Failed to export ONNX model: {e}')

# =============================================================================
# Main Script
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train HA02 attention-convolutional model for channel estimation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--snr', type=int, default=DEFAULT_SNR,
                        choices=sorted(SNR_FOLDER_MAP),
                        help='Channel SNR in dB')
    parser.add_argument('--input-type', type=str, default=DEFAULT_INPUT_TYPE,
                        choices=['ls', 'ls_ori', 'prac', 'li', 'li_ori'],
                        help='Estimate type used for model input')
    parser.add_argument('--epochs', type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help='Adam learning rate')
    parser.add_argument('--train-frac', type=float, default=DEFAULT_TRAIN_FRAC)
    parser.add_argument('--val-frac', type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument('--save-model', action='store_true', default=DEFAULT_SAVE_MODEL)
    parser.add_argument('--save-dir', type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument('--data-root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--test-code', action='store_true')
    
    # Loss config
    parser.add_argument('--loss-type', type=str, default='combined',
                        choices=['combined', 'huber'],
                        help='Loss function type (combined MSE+SSIM or huber)')
    parser.add_argument('--huber-delta', type=float, default=1.0,
                        help='Huber loss transition delta')
    parser.add_argument('--ssim-weight-start', type=float, default=DEFAULT_SSIM_START)
    parser.add_argument('--ssim-weight-end', type=float, default=DEFAULT_SSIM_END)
    parser.add_argument('--standardize', action='store_true',
                        help='Use sample-wise standardization (mean/var) instead of min-max scaling')

    args = parser.parse_args()

    # Visible GPUs
    if args.no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)

    # Load Data
    mat_path = get_data_path(args.data_root, args.snr)
    print(f'[Data] Loading: {mat_path}')
    H_perfect, H_input_pilots, H_li_benchmark_grid, mat_dict, H_perfect_ori = load_mat_data(mat_path, args.input_type)
    N = H_perfect.shape[0]
    
    pilot_cols = mat_dict['pilot_cols'].squeeze() - 1
    pilot_rows = mat_dict['pilot_rows'].squeeze() - 1

    idx_train, idx_val, idx_test = split_indices(N, args.train_frac, args.val_frac)

    if args.test_code:
        idx_train = idx_train[:TEST_CODE_N_TRAIN]
        idx_val   = idx_val[:TEST_CODE_N_VAL]
        idx_test  = idx_test[:TEST_CODE_N_TEST]
        args.epochs = TEST_CODE_EPOCHS

    # Model and Optimizer
    model = HA02Model(num_pilot_elems=H_input_pilots.shape[1], total_grid_elems=14*132)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.5, beta_2=0.9)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Save Setup
    if args.save_dir:
        save_dir = os.path.abspath(args.save_dir)
    else:
        if args.loss_type == 'combined':
            ssim_start_str = str(args.ssim_weight_start).replace('.', '_')
            ssim_end_str = str(args.ssim_weight_end).replace('.', '_')
            suffix = 'standardize' if args.standardize else 'ssim_decay'
            loss_name = f"{suffix}_s{ssim_start_str}_e{ssim_end_str}"
        else:
            huber_delta_str = str(args.huber_delta).replace('.', '_')
            loss_name = f"huber_d{huber_delta_str}"
        save_dir = os.path.join(THIS_DIR, 'trained_models_ha02', f'SNR_{args.snr}dB_{args.input_type}_{loss_name}')
    os.makedirs(save_dir, exist_ok=True)

    lower_range = DEFAULT_LOWER_RANGE
    n_train_batches = len(idx_train) // args.batch_size
    n_val_batches   = len(idx_val) // args.batch_size

    best_val_loss = float('inf')
    best_epoch = 0
    history = {
        'train_loss': [], 'train_mse': [], 'train_ssim': [],
        'val_loss': [], 'val_mse': [], 'val_ssim': [], 'val_nmse': [],
        'ssim_weight_history': []
    }

    print(f'[Train] {args.epochs} epochs  |  {n_train_batches} batches/epoch')
    t_start = time.perf_counter()

    use_huber = (args.loss_type == 'huber')
    huber_delta = tf.constant(args.huber_delta, dtype=tf.float32)

    for epoch in range(args.epochs):
        idx_e = np.random.default_rng(epoch).permutation(idx_train)

        # Decay ssim weight if combined loss
        if args.epochs > 1:
            epoch_ssim_weight = args.ssim_weight_start + (epoch / (args.epochs - 1)) * (args.ssim_weight_end - args.ssim_weight_start)
        else:
            epoch_ssim_weight = args.ssim_weight_start
        history['ssim_weight_history'].append(epoch_ssim_weight)
        ssim_weight_tf = tf.constant(epoch_ssim_weight, dtype=tf.float32)

        # Convert standardize flag to TF constant
        standardize_tf = tf.constant(args.standardize, dtype=tf.bool)

        # Train loop
        ep_train_loss = 0.0
        ep_train_mse  = 0.0
        ep_train_ssim = 0.0
        for b in range(n_train_batches):
            batch_idx = idx_e[b * args.batch_size:(b + 1) * args.batch_size]
            h_p = H_perfect[batch_idx]
            h_i = H_input_pilots[batch_idx]
            x_sc, y_sc, _, _ = preprocess_batch(h_p, h_i, lower_range, standardize=args.standardize)
            
            total_l, mse_l, ssim_l = _train_step(
                model, x_sc, y_sc, optimizer, loss_fn, lower_range, ssim_weight_tf,
                use_huber=use_huber, huber_delta=huber_delta, standardize=standardize_tf
            )
            ep_train_loss += total_l.numpy()
            ep_train_mse  += mse_l.numpy()
            ep_train_ssim += ssim_l.numpy()

        # Val loop
        ep_val_loss = 0.0
        ep_val_mse  = 0.0
        ep_val_ssim = 0.0
        for b in range(n_val_batches):
            batch_idx = idx_val[b * args.batch_size:(b + 1) * args.batch_size]
            h_p = H_perfect[batch_idx]
            h_i = H_input_pilots[batch_idx]
            x_sc, y_sc, _, _ = preprocess_batch(h_p, h_i, lower_range, standardize=args.standardize)
            x_sc = tf.cast(x_sc, tf.float32)
            y_sc = tf.cast(y_sc, tf.float32)
            
            # Predict val
            y_pred_sc = model(x_sc, training=False)
            if use_huber:
                err = tf.abs(y_pred_sc - y_sc)
                h_err = tf.where(err <= huber_delta, 0.5 * tf.square(err), huber_delta * (err - 0.5 * huber_delta))
                total_l = tf.reduce_mean(h_err)
                mse_l = tf.reduce_mean(tf.square(y_pred_sc - y_sc))
                ssim_l = tf.constant(0.0)
            else:
                mse_l = loss_fn(y_sc, y_pred_sc)
                if args.standardize:
                    max_val = tf.reduce_max(y_sc) - tf.reduce_min(y_sc)
                    max_val = tf.maximum(max_val, 1e-8)
                else:
                    max_val = tf.cast(2.0 if lower_range == -1 else 1.0, tf.float32)
                ssim_l = tf.reduce_mean(1.0 - tf_ssim(y_sc, y_pred_sc, max_val=max_val))
                total_l = (1.0 - ssim_weight_tf) * mse_l + ssim_weight_tf * ssim_l
                
            ep_val_loss += total_l.numpy()
            ep_val_mse  += mse_l.numpy()
            ep_val_ssim += ssim_l.numpy()

        # Save history
        history['train_loss'].append(ep_train_loss / max(n_train_batches, 1))
        history['train_mse'].append(ep_train_mse / max(n_train_batches, 1))
        history['train_ssim'].append(ep_train_ssim / max(n_train_batches, 1))
        history['val_loss'].append(ep_val_loss / max(n_val_batches, 1))
        history['val_mse'].append(ep_val_mse / max(n_val_batches, 1))
        history['val_ssim'].append(ep_val_ssim / max(n_val_batches, 1))

        # Checkpoint best val loss
        cur_val_loss = history['val_loss'][-1]
        if cur_val_loss < best_val_loss:
            best_val_loss = cur_val_loss
            best_epoch    = epoch + 1
            if args.save_model:
                export_model_to_onnx(model, os.path.join(save_dir, 'best_model.onnx'), (1, H_input_pilots.shape[1], 2))

        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
            print(f"Epoch {epoch+1:03d}/{args.epochs:03d} | "
                  f"Train Loss: {history['train_loss'][-1]:.6f} (MSE: {history['train_mse'][-1]:.6f}) | "
                  f"Val Loss: {history['val_loss'][-1]:.6f} (MSE: {history['val_mse'][-1]:.6f}) | "
                  f"Best Epoch: {best_epoch:03d} ({best_val_loss:.6f})")

    # Save final model
    if args.save_model:
        export_model_to_onnx(model, os.path.join(save_dir, 'final_model.onnx'), (1, H_input_pilots.shape[1], 2))

    # Save Loss Plots
    save_loss_plot_pdf(history, os.path.join(save_dir))

    # Final Evaluation (Validation + Test sets)
    print('\n[Evaluation] Running final inference on validation & test sets...')
    H_pred_train = infer_channel(model, H_perfect[idx_train], H_input_pilots[idx_train], args.batch_size, lower_range, standardize=args.standardize)
    H_pred_val   = infer_channel(model, H_perfect[idx_val], H_input_pilots[idx_val], args.batch_size, lower_range, standardize=args.standardize)
    H_pred_test  = infer_channel(model, H_perfect[idx_test], H_input_pilots[idx_test], args.batch_size, lower_range, standardize=args.standardize)

    # Compute final metrics
    mmse_train = compute_mmse(H_pred_train, H_perfect[idx_train])
    nmse_train = compute_nmse(H_pred_train, H_perfect[idx_train])
    nmse_train_db = compute_nmse_db(H_pred_train, H_perfect[idx_train])
    ssim_train = compute_ssim_batch(H_pred_train, H_perfect[idx_train])

    mmse_li_benchmark_train = compute_mmse(H_li_benchmark_grid[idx_train], H_perfect[idx_train])
    nmse_li_benchmark_train = compute_nmse(H_li_benchmark_grid[idx_train], H_perfect[idx_train])
    nmse_li_benchmark_train_db = compute_nmse_db(H_li_benchmark_grid[idx_train], H_perfect[idx_train])
    ssim_li_benchmark_train = compute_ssim_batch(H_li_benchmark_grid[idx_train], H_perfect[idx_train])

    mmse_val = compute_mmse(H_pred_val, H_perfect[idx_val])
    nmse_val = compute_nmse(H_pred_val, H_perfect[idx_val])
    nmse_val_db = compute_nmse_db(H_pred_val, H_perfect[idx_val])
    ssim_val = compute_ssim_batch(H_pred_val, H_perfect[idx_val])

    mmse_li_benchmark_val = compute_mmse(H_li_benchmark_grid[idx_val], H_perfect[idx_val])
    nmse_li_benchmark_val = compute_nmse(H_li_benchmark_grid[idx_val], H_perfect[idx_val])
    nmse_li_benchmark_val_db = compute_nmse_db(H_li_benchmark_grid[idx_val], H_perfect[idx_val])
    ssim_li_benchmark_val = compute_ssim_batch(H_li_benchmark_grid[idx_val], H_perfect[idx_val])

    mmse_test = compute_mmse(H_pred_test, H_perfect[idx_test])
    nmse_test = compute_nmse(H_pred_test, H_perfect[idx_test])
    nmse_test_db = compute_nmse_db(H_pred_test, H_perfect[idx_test])
    ssim_test = compute_ssim_batch(H_pred_test, H_perfect[idx_test])

    mmse_li_benchmark_test = compute_mmse(H_li_benchmark_grid[idx_test], H_perfect[idx_test])
    nmse_li_benchmark_test = compute_nmse(H_li_benchmark_grid[idx_test], H_perfect[idx_test])
    nmse_li_benchmark_test_db = compute_nmse_db(H_li_benchmark_grid[idx_test], H_perfect[idx_test])
    ssim_li_benchmark_test = compute_ssim_batch(H_li_benchmark_grid[idx_test], H_perfect[idx_test])

    # ── LMMSE Baseline Estimation ─────────────────────────────────────────────
    print('\n' + '-' * 58)
    print('[LMMSE] Calculating LMMSE baseline estimated channel...')
    has_lmmse = False
    try:
        H_perfect_pilots = H_perfect[:, pilot_rows, pilot_cols]
        H_perfect_vec = H_perfect.reshape(N, -1)
        
        # Complex covariance/correlation matrices
        R_HP = np.matmul(H_perfect_vec.transpose(1, 0).conj(), H_perfect_pilots) / N
        R_PP = np.matmul(H_perfect_pilots.transpose(1, 0).conj(), H_perfect_pilots) / N
        
        noise_var = float(np.mean(np.abs(H_input_pilots - H_perfect_pilots) ** 2))
        
        C = R_PP + noise_var * np.eye(R_PP.shape[0], dtype=np.complex128)
        inv_C = np.linalg.inv(C)
        W = np.matmul(R_HP, inv_C)
        
        H_lmmse_train = np.matmul(H_input_pilots[idx_train], W.T).reshape(-1, 132, 14)
        H_lmmse_val = np.matmul(H_input_pilots[idx_val], W.T).reshape(-1, 132, 14)
        H_lmmse_test = np.matmul(H_input_pilots[idx_test], W.T).reshape(-1, 132, 14)
        
        mmse_lmmse_train = compute_mmse(H_lmmse_train, H_perfect[idx_train])
        nmse_lmmse_train = compute_nmse(H_lmmse_train, H_perfect[idx_train])
        nmse_db_lmmse_train = 10.0 * np.log10(nmse_lmmse_train + 1e-30)
        ssim_lmmse_train = compute_ssim_batch(H_lmmse_train, H_perfect[idx_train])
        
        mmse_lmmse_val = compute_mmse(H_lmmse_val, H_perfect[idx_val])
        nmse_lmmse_val = compute_nmse(H_lmmse_val, H_perfect[idx_val])
        nmse_db_lmmse_val = 10.0 * np.log10(nmse_lmmse_val + 1e-30)
        ssim_lmmse_val = compute_ssim_batch(H_lmmse_val, H_perfect[idx_val])
        
        mmse_lmmse_test = compute_mmse(H_lmmse_test, H_perfect[idx_test])
        nmse_lmmse_test = compute_nmse(H_lmmse_test, H_perfect[idx_test])
        nmse_db_lmmse_test = 10.0 * np.log10(nmse_lmmse_test + 1e-30)
        ssim_lmmse_test = compute_ssim_batch(H_lmmse_test, H_perfect[idx_test])
        
        has_lmmse = True
        print("[LMMSE] LMMSE estimation completed successfully.")
    except Exception as e:
        print(f"[LMMSE Warning] Failed to compute LMMSE baseline: {e}")

    # Save to final_epoch.txt
    txt_path = os.path.join(save_dir, 'final_epoch.txt')
    os.makedirs(os.path.join(save_dir), exist_ok=True)
    try:
        elapsed_total = time.perf_counter() - t_start
        with open(txt_path, 'w') as f:
            f.write("=== FINAL EPOCH EVALUATION RESULTS ===\n")
            f.write(f"SNR (dB):             {args.snr}\n")
            f.write(f"Input Type:           {args.input_type}\n")
            f.write(f"Loss Type:            {args.loss_type}\n")
            f.write(f"Standardize:          {args.standardize}\n")
            f.write(f"Total Execution Time: {elapsed_total:.1f} s\n")
            f.write(f"Best Training Epoch:  {best_epoch}\n\n")
            
            # --- TRAIN ---
            f.write("--- TRAIN SET METRICS ---\n")
            f.write(f"LI Benchmark MMSE:    {mmse_li_benchmark_train:e}\n")
            f.write(f"HA02 Output MMSE:     {mmse_train:e}\n")
            if has_lmmse:
                f.write(f"LMMSE Baseline MMSE:  {mmse_lmmse_train:e}\n")
            f.write(f"LI Benchmark NMSE:    {nmse_li_benchmark_train:e} ({nmse_li_benchmark_train_db:.2f} dB)\n")
            f.write(f"HA02 Output NMSE:     {nmse_train:e} ({nmse_train_db:.2f} dB)\n")
            if has_lmmse:
                f.write(f"LMMSE Baseline NMSE:  {nmse_lmmse_train:e} ({nmse_db_lmmse_train:.2f} dB)\n")
            f.write(f"LI Benchmark SSIM:    {ssim_li_benchmark_train:.4f}\n")
            f.write(f"HA02 Output SSIM:     {ssim_train:.4f}\n")
            if has_lmmse:
                f.write(f"LMMSE Baseline SSIM:  {ssim_lmmse_train:.4f}\n")
            f.write("\n")
            
            # --- VALIDATION ---
            f.write("--- VALIDATION SET METRICS ---\n")
            f.write(f"LI Benchmark MMSE:    {mmse_li_benchmark_val:e}\n")
            f.write(f"HA02 Output MMSE:     {mmse_val:e}\n")
            if has_lmmse:
                f.write(f"LMMSE Baseline MMSE:  {mmse_lmmse_val:e}\n")
            f.write(f"LI Benchmark NMSE:    {nmse_li_benchmark_val:e} ({nmse_li_benchmark_val_db:.2f} dB)\n")
            f.write(f"HA02 Output NMSE:     {nmse_val:e} ({nmse_val_db:.2f} dB)\n")
            if has_lmmse:
                f.write(f"LMMSE Baseline NMSE:  {nmse_lmmse_val:e} ({nmse_db_lmmse_val:.2f} dB)\n")
            f.write(f"LI Benchmark SSIM:    {ssim_li_benchmark_val:.4f}\n")
            f.write(f"HA02 Output SSIM:     {ssim_val:.4f}\n")
            if has_lmmse:
                f.write(f"LMMSE Baseline SSIM:  {ssim_lmmse_val:.4f}\n")
            f.write("\n")
            
            # --- TEST ---
            f.write("--- TEST SET METRICS ---\n")
            f.write(f"LI Benchmark MMSE:    {mmse_li_benchmark_test:e}\n")
            f.write(f"HA02 Output MMSE:     {mmse_test:e}\n")
            if has_lmmse:
                f.write(f"LMMSE Baseline MMSE:  {mmse_lmmse_test:e}\n")
            f.write(f"LI Benchmark NMSE:    {nmse_li_benchmark_test:e} ({nmse_li_benchmark_test_db:.2f} dB)\n")
            f.write(f"HA02 Output NMSE:     {nmse_test:e} ({nmse_test_db:.2f} dB)\n")
            if has_lmmse:
                f.write(f"LMMSE Baseline NMSE:  {nmse_lmmse_test:e} ({nmse_db_lmmse_test:.2f} dB)\n")
            f.write(f"LI Benchmark SSIM:    {ssim_li_benchmark_test:.4f}\n")
            f.write(f"HA02 Output SSIM:     {ssim_test:.4f}\n")
            if has_lmmse:
                f.write(f"LMMSE Baseline SSIM:  {ssim_lmmse_test:.4f}\n")
        print(f"[Save] Final epoch text report -> {txt_path}")
    except Exception as e:
         print(f"[Save Warning] Failed to write final_epoch.txt report: {e}")

    # Save to evaluation_results.mat
    eval_path = os.path.join(save_dir, 'evaluation_results.mat')
    eval_dict = {
        'mmse_train': mmse_train, 'nmse_train': nmse_train, 'nmse_train_db': nmse_train_db, 'ssim_train': ssim_train,
        'mmse_li_benchmark_train': mmse_li_benchmark_train, 'nmse_li_benchmark_train': nmse_li_benchmark_train, 'nmse_li_benchmark_train_db': nmse_li_benchmark_train_db, 'ssim_li_benchmark_train': ssim_li_benchmark_train,
        'mmse_val': mmse_val, 'nmse_val': nmse_val, 'nmse_val_db': nmse_val_db, 'ssim_val': ssim_val,
        'mmse_li_benchmark_val': mmse_li_benchmark_val, 'nmse_li_benchmark_val': nmse_li_benchmark_val, 'nmse_li_benchmark_val_db': nmse_li_benchmark_val_db, 'ssim_li_benchmark_val': ssim_li_benchmark_val,
        'mmse_test': mmse_test, 'nmse_test': nmse_test, 'nmse_test_db': nmse_test_db, 'ssim_test': ssim_test,
        'mmse_li_benchmark_test': mmse_li_benchmark_test, 'nmse_li_benchmark_test': nmse_li_benchmark_test, 'nmse_li_benchmark_test_db': nmse_li_benchmark_test_db, 'ssim_li_benchmark_test': ssim_li_benchmark_test,
        'snr': args.snr, 'input_type': args.input_type, 'standardize': args.standardize, 'best_epoch': best_epoch
    }
    if has_lmmse:
        eval_dict.update({
            'mmse_lmmse_train': mmse_lmmse_train, 'nmse_lmmse_train': nmse_lmmse_train, 'nmse_db_lmmse_train': nmse_db_lmmse_train, 'ssim_lmmse_train': ssim_lmmse_train,
            'mmse_lmmse_val': mmse_lmmse_val, 'nmse_lmmse_val': nmse_lmmse_val, 'nmse_db_lmmse_val': nmse_db_lmmse_val, 'ssim_lmmse_val': ssim_lmmse_val,
            'mmse_lmmse_test': mmse_lmmse_test, 'nmse_lmmse_test': nmse_lmmse_test, 'nmse_db_lmmse_test': nmse_db_lmmse_test, 'ssim_lmmse_test': ssim_lmmse_test,
        })
    scipy.io.savemat(eval_path, eval_dict)
    print(f"[Save] Evaluation results -> {eval_path}")

    # Save the test set channel grids for BER / metrics visualization
    test_grids_path = os.path.join(save_dir, 'testChannel.mat')
    test_grids_dict = {
        'H_original_test': H_perfect_ori[idx_test],      # Original channel before Doppler comp
        'H_perfect_test': H_perfect[idx_test],            # Effective channel after Doppler comp
        'H_LS_test': H_input_pilots[idx_test],            # Pilot sequences (LS channel estimates)
        'pilot_rows': pilot_rows + 1,                     # 1-indexed row coordinates of pilots for MATLAB
        'pilot_cols': pilot_cols + 1,                     # 1-indexed column coordinates of pilots for MATLAB
        'H_LI_test': H_li_benchmark_grid[idx_test],       # Benchmark LI channel grid
        'H_output_test': H_pred_test,                     # Model output channel grid
        'snr': args.snr
    }
    scipy.io.savemat(test_grids_path, test_grids_dict)
    print(f"[Save] Saved test channel grids MAT file -> {test_grids_path}")

    # Copy readme*.md from dataset folder to results directory
    try:
        import shutil
        import glob
        snr_folder_name = SNR_FOLDER_MAP.get(args.snr, f'{args.snr}dB')
        md_pattern = os.path.join(PROJECT_ROOT, 'generatedChan', 'OpenNTN', DATA_FOLDER_NAME, snr_folder_name, 'readme*.md')
        md_matches = glob.glob(md_pattern)
        target_dir = os.path.join(save_dir)
        if md_matches:
            md_src = md_matches[0]
            shutil.copy(md_src, target_dir)
            print(f"[Save] Copied dataset readme ({os.path.basename(md_src)}) to: {target_dir}")
        else:
            print(f"[Save Warning] Metadata readme matching '{md_pattern}' not found.")
    except Exception as e:
        print(f"[Save Warning] Failed to copy metadata readme: {e}")

    # Save plots
    if len(idx_test) > 0:
        save_channel_plots_pdf(
            H_perfect[idx_test[0]],
            H_input_pilots[idx_test[0]],
            H_pred_test[0],
            pilot_rows, pilot_cols,
            args.input_type,
            os.path.join(save_dir),
            prefix='test'
        )
    if len(idx_train) > 0:
        save_channel_plots_pdf(
            H_perfect[idx_train[0]],
            H_input_pilots[idx_train[0]],
            H_pred_train[0],
            pilot_rows, pilot_cols,
            args.input_type,
            os.path.join(save_dir),
            prefix='train'
        )

    print(f'[Done] Finished training and evaluation. Results saved in: {save_dir}')

if __name__ == '__main__':
    main()
