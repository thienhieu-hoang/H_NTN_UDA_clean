"""
Single-Dataset Attention-based Channel Estimator with Multi-Scale U-Net Refinement (OpenNTN)
=============================================================================================
Stage 1: Coarse reconstruction via Transformer-Encoder + Dense Upsampling (HA02).
Stage 2: Multi-Scale U-Net Refiner with hierarchical downsampling, global bottleneck, 
         and skip connections to predict fine-grained channel residual.
Combined: H_refined = H_coarse + H_unet_refine.
=============================================================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import os
import sys
import time
import argparse

# ── Third-party ──────────────────────────────────────────────────────────────
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

    if os.path.basename(this_dir).lower() == 'ls_attention_plus':
        project_root = os.path.abspath(os.path.join(this_dir, '..', '..'))
    else:
        project_root = os.path.abspath(os.path.join(this_dir, '..'))

    for p in [project_root,
              os.path.join(project_root, 'single_dataset'),
              os.path.join(project_root, 'Domain_Adversarial', 'helper'),
              os.path.join(project_root, 'JMMD', 'helper')]:
        if p not in sys.path:
            sys.path.insert(0, p)
    return this_dir, project_root

THIS_DIR, PROJECT_ROOT = _setup_paths()

from utils import standardizeScaler_ha02, deStandardize_ha02

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
    def __init__(self, num_pilot_elems=88, total_grid_elems=1848, n_filter=2, **kwargs):
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
        self.conv_out = tf.keras.layers.Conv2D(filters=1, kernel_size=(2, 2), padding='same', name="conv_out")

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
        
        out = self.conv_out(h2_upsampled)
        out = tf.squeeze(out, axis=-1)
        out_grid = tf.reshape(out, [B, self.num_subcarriers, 14, 2])
        return out_grid

# =============================================================================
# 3. MULTI-SCALE U-NET REFINEMENT BLOCK
# =============================================================================
class UNetRefineBlock(tf.keras.layers.Layer):
    """
    Multi-Scale U-Net Refiner:
    - Level 1: [B, 132, 14, base_filters]
    - Level 2: Downsampled by (2, 2) -> [B, 66, 7, base_filters*2]
    - Bottleneck: Downsampled by (2, 1) -> [B, 33, 7, base_filters*4]
    - Upsampling with skip connections back to [B, 132, 14, 2]
    """
    def __init__(self, base_filters=32, **kwargs):
        super(UNetRefineBlock, self).__init__(**kwargs)
        self.base_filters = base_filters
        
        # Level 1 Encoder (132 x 14)
        self.enc1_1 = tf.keras.layers.Conv2D(base_filters, (3, 3), padding='same', name="u_enc1_1")
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="u_ln1")
        self.enc1_2 = tf.keras.layers.Conv2D(base_filters, (3, 3), padding='same', name="u_enc1_2")
        self.down1  = tf.keras.layers.Conv2D(base_filters * 2, (3, 3), strides=(2, 2), padding='same', name="u_down1")
        
        # Level 2 Encoder (66 x 7)
        self.enc2_1 = tf.keras.layers.Conv2D(base_filters * 2, (3, 3), padding='same', name="u_enc2_1")
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="u_ln2")
        self.enc2_2 = tf.keras.layers.Conv2D(base_filters * 2, (3, 3), padding='same', name="u_enc2_2")
        self.down2  = tf.keras.layers.Conv2D(base_filters * 4, (3, 3), strides=(2, 1), padding='same', name="u_down2")
        
        # Bottleneck (33 x 7)
        self.bneck1 = tf.keras.layers.Conv2D(base_filters * 4, (3, 3), padding='same', name="u_bneck1")
        self.ln_bn  = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="u_ln_bn")
        self.bneck2 = tf.keras.layers.Conv2D(base_filters * 4, (3, 3), padding='same', name="u_bneck2")
        
        # Level 2 Decoder
        self.up2    = tf.keras.layers.Conv2DTranspose(base_filters * 2, (3, 3), strides=(2, 1), padding='same', name="u_up2")
        self.dec2_1 = tf.keras.layers.Conv2D(base_filters * 2, (3, 3), padding='same', name="u_dec2_1")
        self.dec2_2 = tf.keras.layers.Conv2D(base_filters * 2, (3, 3), padding='same', name="u_dec2_2")
        
        # Level 1 Decoder
        self.up1    = tf.keras.layers.Conv2DTranspose(base_filters, (3, 3), strides=(2, 2), padding='same', name="u_up1")
        self.dec1_1 = tf.keras.layers.Conv2D(base_filters, (3, 3), padding='same', name="u_dec1_1")
        self.dec1_2 = tf.keras.layers.Conv2D(base_filters, (3, 3), padding='same', name="u_dec1_2")
        
        # Final projection to Delta H
        self.out_conv = tf.keras.layers.Conv2D(2, (3, 3), padding='same', name="u_out_conv")

    def call(self, h_coarse, training=False, return_bottleneck=False):
        # Level 1
        x1 = tf.nn.leaky_relu(self.ln1(self.enc1_1(h_coarse)))
        x1 = tf.nn.leaky_relu(self.enc1_2(x1))  # [B, 132, 14, base_filters]
        
        # Level 2
        d1 = tf.nn.leaky_relu(self.down1(x1))
        x2 = tf.nn.leaky_relu(self.ln2(self.enc2_1(d1)))
        x2 = tf.nn.leaky_relu(self.enc2_2(x2))  # [B, 66, 7, base_filters*2]
        
        # Bottleneck
        d2 = tf.nn.leaky_relu(self.down2(x2))
        bn = tf.nn.leaky_relu(self.ln_bn(self.bneck1(d2)))
        bn = tf.nn.leaky_relu(self.bneck2(bn))  # [B, 33, 7, base_filters*4]
        
        # Decoder 2
        u2 = tf.nn.leaky_relu(self.up2(bn))     # [B, 66, 7, base_filters*2]
        c2 = tf.concat([u2, x2], axis=-1)
        y2 = tf.nn.leaky_relu(self.dec2_1(c2))
        y2 = tf.nn.leaky_relu(self.dec2_2(y2))
        
        # Decoder 1
        u1 = tf.nn.leaky_relu(self.up1(y2))     # [B, 132, 14, base_filters]
        c1 = tf.concat([u1, x1], axis=-1)
        y1 = tf.nn.leaky_relu(self.dec1_1(c1))
        y1 = tf.nn.leaky_relu(self.dec1_2(y1))
        
        delta_h = self.out_conv(y1)             # [B, 132, 14, 2]
        
        if return_bottleneck:
            return delta_h, bn
        return delta_h

# =============================================================================
# 4. COMPLETE HA02 U-NET REFINEMENT MODEL
# =============================================================================
class HA02UNetRefineModel(tf.keras.Model):
    def __init__(self, num_pilot_elems=88, total_grid_elems=1848, num_channels=2, 
                 num_heads=2, n_filter=2, refine_filters=32, **kwargs):
        super(HA02UNetRefineModel, self).__init__(**kwargs)
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
        self.unet_refiner = UNetRefineBlock(base_filters=refine_filters)

    def call(self, inputs, training=False, return_intermediate=False):
        encoder_out = self.encoder(inputs)
        h_coarse = self.decoder(encoder_out, training=training)
        
        if return_intermediate:
            delta_h, bn = self.unet_refiner(h_coarse, training=training, return_bottleneck=True)
            h_refined = h_coarse + delta_h
            return h_refined, h_coarse, bn
        else:
            delta_h = self.unet_refiner(h_coarse, training=training)
            h_refined = h_coarse + delta_h
            return h_refined

# =============================================================================
# 5. HUBER LOSS FUNCTION
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
    if not os.path.exists(base_dir):
        return None

    primary_dataset_names = [
        'matlabNTN.mat',
        'channel_dur_randomizedUE.mat',
        'channel_dur.mat',
        'channel_sur.mat',
        'channel.mat',
        'dataset.mat',
        'data.mat',
    ]

    for name in primary_dataset_names:
        candidate = os.path.join(base_dir, name)
        if os.path.isfile(candidate):
            return candidate

    if os.path.isdir(base_dir):
        files_in_dir = os.listdir(base_dir)
        for name in primary_dataset_names:
            for f in files_in_dir:
                if f.lower() == name.lower() and os.path.isfile(os.path.join(base_dir, f)):
                    return os.path.join(base_dir, f)

    excluded_prefixes = (
        'testChannel', 'inferredChannel', 'training_history',
        'channel_grids', 'synthesized', 'evaluation_results',
        'sample_', 'extracted_features', 'loss_', 'nmse_', 'best_', 'final_'
    )

    for root, _, files in os.walk(base_dir):
        for f in sorted(files):
            if f.endswith('.mat') and not f.startswith(excluded_prefixes):
                return os.path.join(root, f)

    return None

def get_data_path(data_root: str, snr: int, is_test_code: bool = False) -> str:
    this_dir, project_root = _setup_paths()

    candidate_roots = []
    if data_root:
        if os.path.isabs(data_root):
            candidate_roots.append(data_root)
        else:
            candidate_roots.append(os.path.abspath(data_root))
            candidate_roots.append(os.path.join(project_root, data_root))
            candidate_roots.append(os.path.join(this_dir, data_root))
            
            for parent in [os.path.join(project_root, 'generatedChan', 'MATLAB'),
                           os.path.join(project_root, 'generatedChan', 'OpenNTN')]:
                if os.path.isdir(parent):
                    base_name = os.path.basename(data_root.rstrip('\\/'))
                    for entry in os.listdir(parent):
                        if base_name in entry:
                            candidate_roots.append(os.path.join(parent, entry))

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
        for snr_var in snr_variations:
            snr_dir = os.path.join(root, snr_var)
            if os.path.isdir(snr_dir):
                mat_file = find_channel_mat_file(snr_dir)
                if mat_file:
                    return mat_file

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

@tf.function
def _train_step(model, x_scaled, y_scaled, optimizer, loss_fn, lower_range, ssim_weight, 
                use_huber=False, huber_delta=1.0, standardize=False):
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
    try:
        import tf2onnx
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tf2onnx'])
        import tf2onnx

    try:
        spec = (tf.TensorSpec(input_shape, tf.float32, name='input_channel'),)
        model_proto, _ = tf2onnx.convert.from_keras(
            model, input_signature=spec, output_path=save_path)
        print(f'[ONNX Export] Saved ONNX model (architecture + weights) -> {save_path}')
    except Exception as e:
        print(f'[ONNX Export Warning] ONNX conversion failed: {e}')

# =============================================================================
# MAIN SCRIPT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train HA02 Attention with Multi-Scale U-Net Refiner.')
    parser.add_argument('--snr', type=int, default=DEFAULT_SNR, help='Target SNR in dB')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR, help='Learning rate')
    parser.add_argument('--train-frac', type=float, default=DEFAULT_TRAIN_FRAC, help='Train fraction')
    parser.add_argument('--val-frac', type=float, default=DEFAULT_VAL_FRAC, help='Val fraction')
    parser.add_argument('--input-type', type=str, default=DEFAULT_INPUT_TYPE, choices=['ls', 'ls_ori'])
    parser.add_argument('--test-code', action='store_true', help='Smoke test mode')
    parser.add_argument('--save-model', action='store_true', help='Save best model to ONNX')
    parser.add_argument('--save-dir', type=str, default=DEFAULT_SAVE_DIR, help='Custom save directory')
    parser.add_argument('--data-root', type=str, default=DEFAULT_DATA_ROOT, help='Dataset root path')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--loss-type', type=str, default='combined', choices=['combined', 'huber'])
    parser.add_argument('--huber-delta', type=float, default=1.0)
    parser.add_argument('--ssim-weight-start', type=float, default=DEFAULT_SSIM_START)
    parser.add_argument('--ssim-weight-end', type=float, default=DEFAULT_SSIM_END)
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--refine-filters', type=int, default=32, help='Base filters in U-Net refiner')

    args = parser.parse_args()

    if args.no_gpu:
        tf.config.set_visible_devices([], 'GPU')
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)

    mat_path = get_data_path(args.data_root, args.snr)
    print(f'[Data] Loading: {mat_path}')
    H_perfect, H_input_pilots, H_li_benchmark_grid, mat_dict, H_perfect_ori = load_mat_data(mat_path, args.input_type)
    N = H_perfect.shape[0]

    idx_train, idx_val, idx_test = split_indices(N, args.train_frac, args.val_frac)

    if args.test_code:
        idx_train = idx_train[:TEST_CODE_N_TRAIN]
        idx_val   = idx_val[:TEST_CODE_N_VAL]
        idx_test  = idx_test[:TEST_CODE_N_TEST]
        args.epochs = TEST_CODE_EPOCHS

    model = HA02UNetRefineModel(
        num_pilot_elems=H_input_pilots.shape[1], 
        total_grid_elems=14*132,
        refine_filters=args.refine_filters
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.5, beta_2=0.9)
    loss_fn = tf.keras.losses.MeanSquaredError()

    if args.save_dir:
        save_dir = os.path.abspath(args.save_dir)
    else:
        suffix = 'standardize' if args.standardize else 'minmax'
        save_dir = os.path.join(THIS_DIR, 'trained_models_unet_refine', f'SNR_{args.snr}dB_{args.input_type}_{suffix}')
    os.makedirs(save_dir, exist_ok=True)

    lower_range = DEFAULT_LOWER_RANGE
    n_train_batches = len(idx_train) // args.batch_size
    n_val_batches   = len(idx_val) // args.batch_size

    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'train_mse': [], 'train_ssim': [], 'val_loss': [], 'val_mse': [], 'val_ssim': []}

    print(f'[Train] {args.epochs} epochs  |  {n_train_batches} batches/epoch')
    use_huber = (args.loss_type == 'huber')
    huber_delta = tf.constant(args.huber_delta, dtype=tf.float32)

    for epoch in range(args.epochs):
        idx_e = np.random.default_rng(epoch).permutation(idx_train)

        if args.epochs > 1:
            epoch_ssim_weight = args.ssim_weight_start + (epoch / (args.epochs - 1)) * (args.ssim_weight_end - args.ssim_weight_start)
        else:
            epoch_ssim_weight = args.ssim_weight_start
        ssim_weight_tf = tf.constant(epoch_ssim_weight, dtype=tf.float32)
        standardize_tf = tf.constant(args.standardize, dtype=tf.bool)

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

        history['train_loss'].append(ep_train_loss / max(n_train_batches, 1))
        history['train_mse'].append(ep_train_mse / max(n_train_batches, 1))
        history['train_ssim'].append(ep_train_ssim / max(n_train_batches, 1))
        history['val_loss'].append(ep_val_loss / max(n_val_batches, 1))
        history['val_mse'].append(ep_val_mse / max(n_val_batches, 1))
        history['val_ssim'].append(ep_val_ssim / max(n_val_batches, 1))

        cur_val_loss = history['val_loss'][-1]
        if cur_val_loss < best_val_loss:
            best_val_loss = cur_val_loss
            best_epoch    = epoch + 1
            if args.save_model:
                export_model_to_onnx(model, os.path.join(save_dir, 'best_model.onnx'), (1, H_input_pilots.shape[1], 2))

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
            print(f'Epoch {epoch+1:03d}/{args.epochs:03d} | Train: {history["train_loss"][-1]:.4e} | Val: {cur_val_loss:.4e} (Best: {best_val_loss:.4e} @ ep {best_epoch})')

    if args.save_model:
        export_model_to_onnx(model, os.path.join(save_dir, 'final_model.onnx'), (1, H_input_pilots.shape[1], 2))

    # Evaluate on test split
    print('[Test] Evaluating final model on test split ...')
    h_p_test = H_perfect[idx_test]
    h_i_test = H_input_pilots[idx_test]
    H_pred_test = infer_channel(model, h_p_test, h_i_test, batch_size=args.batch_size, lower_range=lower_range, standardize=args.standardize)
    test_nmse = compute_nmse(H_pred_test, h_p_test)
    test_nmse_db = compute_nmse_db(H_pred_test, h_p_test)
    print(f'[Test Results] NMSE: {test_nmse:.6f} | NMSE (dB): {test_nmse_db:.2f} dB')

if __name__ == '__main__':
    main()
