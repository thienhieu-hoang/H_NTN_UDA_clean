"""
Single-Dataset CNN Channel Estimator with Cross Self-Attention and Dynamic MSE & SSIM Loss Schedule (OpenNTN)
=====================================================================================
Train, evaluate, and test a DnCNN_ResNet_CrossAttention model on ONE SNR split of the OpenNTN dataset.
All data (train / val / test) come from the *same* SNR folder.

Features of this script:
1. Model Architecture:
   - Uses `DnCNN_ResNet_CrossAttention` which implements Cross Self-Attention at the middle layer.
     The Query is projected from the full time-frequency grid, while Key and Value are projected 
     from the features gathered at the sparse pilot coordinates (recovering details from the exact pilots).
   - The final output layer has no activation function (completely linear output) allowing values
     to span outside the scaling range.
2. Normalization Options:
   - Sample-wise min-max scaling to [-1, 1] (default).
   - Sample-wise standardization (mean/std normalization) via `--standardize` flag.
3. Preprocessing:
   - If input is H_LI: clipping extrapolation elements (outside pilot boundaries) using the min and max
     values of pilot positions to reduce interpolation boundary errors.
4. Loss Schedule:
   - Starts with a large SSIM weight to learn coarse layout structures, then decays it linearly 
     to let MSE dominate in the later epochs.
====================================================================================
Usage
-----
    python train_DnCNN_CrossAttention.py --snr 10 --ssim-weight-start 0.5 --ssim-weight-end 0.05

    # Example for running a quick smoke test
    python train_DnCNN_CrossAttention.py --snr 10 --test-code --save-model

    # Example for linear interpolation input with extrapolation clipping and standardization
    python train_DnCNN_CrossAttention.py --snr 10 --input-type li --clip-extrap --standardize --save-model
"""

# -- Standard library --------------------------------------------------------
import os
import sys
import time
import argparse

# -- Third-party --------------------------------------------------------------
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
# (edit here when running directly in an IDE; CLI arguments take precedence)
# ============================================================================
DEFAULT_SNR         = 10           # dB
DEFAULT_INPUT_TYPE  = 'li'       # 'prac' or 'li'
DEFAULT_EPOCHS      = 200
DEFAULT_BATCH_SIZE  = 16
DEFAULT_LR          = 1e-4
DEFAULT_TRAIN_FRAC  = 0.70
DEFAULT_VAL_FRAC    = 0.15
DEFAULT_LOWER_RANGE = -1           # minmax scaling range: -1 -> [-1, 1]
DEFAULT_SSIM_START  = 0.95         # Initial SSIM weight (MSE weight = 1 - w)
DEFAULT_SSIM_END    = 0.05         # Final SSIM weight
# Quick-test mode: tiny subset + few epochs (activated by --test-code)
TEST_CODE_N_TRAIN   = 48           # samples used for training in test-code mode
TEST_CODE_N_VAL     = 16           # samples used for validation
TEST_CODE_N_TEST    = 16           # samples used for test
TEST_CODE_EPOCHS    = 5            # epochs to run in test-code mode
DEFAULT_SAVE_MODEL  = False
DEFAULT_SAVE_DIR    = ''           # auto = ./trained_models/SNR_{snr}dB_{input_type}_decay
DEFAULT_DATA_ROOT   = ''           # auto-detected relative to this script
DEFAULT_N_BLOCKS    = 4
DEFAULT_CLIP_EXTRAP = False
# ============================================================================


# ----------------------------------------------------------------------------
# Path setup  (adds project root + JMMD/Domain_Adversarial helpers to sys.path)
# ----------------------------------------------------------------------------
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

# Lazy import of JMMD utilities (requires project_root on sys.path)
from utils_GAN import DnCNN_ResNet_CrossAttention                     # JMMD/helper/utils_GAN.py
from utils import minmaxScaler, deMinMax, complx2real, standardizeScaler, deStandardize      # Domain_Adversarial/helper/utils.py


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
DATA_FOLDER_NAME = 'DUR200ns_27G_600km_r15km_20to30mps'

SNR_FOLDER_MAP = {
    -10: '-10dB',
    -5:  '-5dB',
    0:   '0dB',
    5:   '5dB',
    10:  '10dB',
    15:  '15dB',
}


def find_any_mat_file(base_dir: str) -> str:
    """Recursively search base_dir for the first valid .mat file."""
    if not os.path.exists(base_dir):
        return None
    for root, _, files in os.walk(base_dir):
        for f in sorted(files):
            if f.endswith('.mat'):
                return os.path.join(root, f)
    return None


def get_data_path(data_root: str, snr: int, is_test_code: bool = False) -> str:
    """
    Return the absolute path to the .mat file for the requested SNR.
    """
    openntn_root = os.path.join(PROJECT_ROOT, 'generatedChan', 'OpenNTN')
    snr_folder   = SNR_FOLDER_MAP.get(snr, f'{snr}dB')

    if not data_root:
        target_root = os.path.join(openntn_root, DATA_FOLDER_NAME)
    else:
        target_root = data_root

    mat_path = os.path.join(target_root, snr_folder, 'channel_dur_randomizedUE.mat')

    # Dynamic fallback if configured file does not exist
    if not os.path.isfile(mat_path):
        fallback = find_any_mat_file(target_root) or find_any_mat_file(openntn_root)
        if fallback:
            print(f'[Data Fallback] Target path not found ({mat_path}).\n'
                  f'  Dynamically using available .mat file -> {fallback}')
            return fallback
        raise FileNotFoundError(
            f'Data file not found at:\n  {mat_path}\n'
            f'  and no alternative .mat files found under: {openntn_root}')

    return mat_path


def load_mat_data(mat_path: str, input_type: str):
    """
    Load a .mat file (supports both v7 and v7.3 HDF5 formats) and return numpy arrays.
    """
    try:
        # Try loading with scipy.io.loadmat (v7 or older)
        mat = scipy.io.loadmat(mat_path)
        is_hdf5 = False
    except NotImplementedError:
        # Fallback to h5py (v7.3)
        import h5py
        is_hdf5 = True
        
    if is_hdf5:
        # Helper to convert HDF5 real/imag struct to complex numpy array
        def h5_to_complex(val):
            if isinstance(val, np.ndarray) and val.dtype.names is not None:
                if 'real' in val.dtype.names and 'imag' in val.dtype.names:
                    return val['real'] + 1j * val['imag']
            return val

        with h5py.File(mat_path, 'r') as f:
            # 1. H_perfect key resolution
            perf_key = None
            if 'H_perfect' in f:
                perf_key = 'H_perfect'
            elif 'H_perfect_ori' in f:
                perf_key = 'H_perfect_ori'
            else:
                for k in f.keys():
                    if not k.startswith('#') and len(f[k].shape) == 3:
                        perf_key = k
                        break
            if perf_key is None:
                raise KeyError("H_perfect channel key not found in HDF5 file.")
                
            H_perfect = h5_to_complex(f[perf_key][()])
            if H_perfect.ndim == 3:
                # Swapped dimensions in HDF5: (N, 14, 132) -> transpose to (N, 132, 14)
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
            
            N_samples, n_subc, n_symb = H_perfect.shape
            
            # Reconstruct dictionary of other variables for fallback
            mat_dict = {}
            for k in f.keys():
                if k.startswith('#'):
                    continue
                mat_dict[k] = h5_to_complex(f[k][()])
                # Transpose pilot coords if they are saved as column/row matrices
                if k in ['pilot_rows', 'pilot_cols']:
                    mat_dict[k] = np.squeeze(mat_dict[k])
                # If they are other matrices of size (N, 14, 132), transpose them to (N, 132, 14)
                elif isinstance(mat_dict[k], np.ndarray) and mat_dict[k].ndim == 3:
                    if mat_dict[k].shape[1] == 14 and mat_dict[k].shape[2] == 132:
                        mat_dict[k] = np.transpose(mat_dict[k], (0, 2, 1))

            # 2. H_input key resolution
            if input_type in ['ls', 'ls_ori']:
                ls_key = 'H_ls_pilots_ori' if input_type == 'ls_ori' and 'H_ls_pilots_ori' in mat_dict else 'H_ls_pilots'
                if ls_key not in mat_dict:
                    for alt in ['H_ls_pilots', 'H_ls_pilots_ori', 'H_LS_comp', 'H_LS_full']:
                        if alt in mat_dict:
                            ls_key = alt
                            break
                            
                if ls_key not in mat_dict or 'pilot_rows' not in mat_dict or 'pilot_cols' not in mat_dict:
                    raise KeyError(f"Unable to reconstruct sparse LS grid. Required keys '{ls_key}', "
                                   f"'pilot_rows', 'pilot_cols' not found in {mat_path}")
                                   
                H_pilots   = mat_dict[ls_key]
                pilot_rows = mat_dict['pilot_rows'].astype(int) - 1
                pilot_cols = mat_dict['pilot_cols'].astype(int) - 1
                
                if H_pilots.ndim == 2:
                    if H_pilots.shape[0] != N_samples and H_pilots.shape[1] == N_samples:
                        H_pilots = H_pilots.T
                        
                H_input = np.zeros((N_samples, n_subc, n_symb), dtype=np.complex64)
                for i in range(N_samples):
                    H_input[i, pilot_rows, pilot_cols] = H_pilots[i, :]
                input_key = f"{ls_key} (sparse HDF5)"
            else:
                input_key_map = {'prac': 'H_prac', 'li': 'H_li', 'li_ori': 'H_li_ori'}
                input_key = input_key_map.get(input_type, 'H_li')
                if input_key not in mat_dict:
                    for alt in [input_key, 'H_li', 'H_li_ori', 'H_prac']:
                        if alt in mat_dict:
                            input_key = alt
                            break
                H_input = mat_dict[input_key]
                
            H_li_benchmark_grid = mat_dict.get('H_li', mat_dict.get('H_li_ori', mat_dict.get('H_perfect')))
            if H_li_benchmark_grid.ndim == 2:
                H_li_benchmark_grid = H_perfect
                
    else:
        # 1. H_perfect key resolution
        if 'H_perfect' in mat and mat['H_perfect'].size > 0:
            H_perfect = mat['H_perfect'].T
        elif 'H_perfect_ori' in mat and mat['H_perfect_ori'].size > 0:
            H_perfect = mat['H_perfect_ori'].T
        else:
            for k in mat.keys():
                if not k.startswith('__') and isinstance(mat[k], np.ndarray) and mat[k].ndim == 3:
                    H_perfect = mat[k].T
                    break

        # Load H_perfect_ori
        H_perfect_ori = None
        for key in ['H_perfect_ori', 'H_perfect_original', 'H_true_ori', 'H_ori']:
            if key in mat:
                H_perfect_ori = mat[key].T
                break
        if H_perfect_ori is None:
            H_perfect_ori = H_perfect

        N_samples, n_subc, n_symb = H_perfect.shape

        # Create mat_dict to match
        mat_dict = mat

        # 2. H_input key resolution
        if input_type in ['ls', 'ls_ori']:
            ls_key = 'H_ls_pilots_ori' if input_type == 'ls_ori' and 'H_ls_pilots_ori' in mat else 'H_ls_pilots'
            if ls_key not in mat:
                for alt in ['H_ls_pilots', 'H_ls_pilots_ori', 'H_LS_comp', 'H_LS_full']:
                    if alt in mat and mat[alt].size > 0:
                        ls_key = alt
                        break

            if ls_key not in mat or 'pilot_rows' not in mat or 'pilot_cols' not in mat:
                raise KeyError(f"Unable to reconstruct sparse LS grid. Required keys '{ls_key}', "
                               f"'pilot_rows', 'pilot_cols' not found in {mat_path}")

            H_pilots   = mat[ls_key]
            pilot_rows = np.squeeze(mat['pilot_rows']).astype(int) - 1
            pilot_cols = np.squeeze(mat['pilot_cols']).astype(int) - 1

            if H_pilots.shape[0] != N_samples and H_pilots.shape[1] == N_samples:
                H_pilots = H_pilots.T

            H_input = np.zeros((N_samples, n_subc, n_symb), dtype=np.complex64)
            for i in range(N_samples):
                H_input[i, pilot_rows, pilot_cols] = H_pilots[i, :]
            input_key = f"{ls_key} (sparse 2D grid)"
        else:
            input_key_map = {'prac': 'H_prac', 'li': 'H_li', 'li_ori': 'H_li_ori'}
            input_key = input_key_map.get(input_type, 'H_li')
            if input_key not in mat or mat[input_key].size == 0:
                for alt in [input_key, 'H_li', 'H_li_ori', 'H_prac']:
                    if alt in mat and isinstance(mat[alt], np.ndarray) and mat[alt].size > 0:
                        input_key = alt
                        break
            H_input = mat[input_key].T

        H_li_benchmark_grid = mat.get('H_li', mat.get('H_li_ori', mat['H_perfect'])).T
        if H_li_benchmark_grid.ndim == 2:
            H_li_benchmark_grid = H_perfect

    H_perfect = H_perfect.astype(np.complex64)
    H_input   = H_input.astype(np.complex64)
    print(f'  Loaded H_perfect {H_perfect.shape}  |  H_input ({input_key}) {H_input.shape}')
    return H_perfect, H_input, mat_dict, H_perfect_ori, H_li_benchmark_grid


def split_indices(N: int, train_frac: float, val_frac: float, seed: int = 1234):
    """Return reproducible (train, val, test) index arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


# ----------------------------------------------------------------------------
# Helpers: structured dtype required by complx2real()
# ----------------------------------------------------------------------------
_STRUCT_DTYPE = np.dtype([('real', np.float32), ('imag', np.float32)])


def to_structured(H: np.ndarray) -> np.ndarray:
    """Convert a plain complex array [N, S, T] to structured dtype."""
    out = np.empty(H.shape, dtype=_STRUCT_DTYPE)
    out['real'] = H.real.astype(np.float32)
    out['imag'] = H.imag.astype(np.float32)
    return out




# ----------------------------------------------------------------------------
# Pre-processing
# ----------------------------------------------------------------------------
def preprocess_batch(H_perf_batch: np.ndarray, H_in_batch: np.ndarray,
                     lower_range: int, clip_extrap: bool = False,
                     pilot_bounds: tuple = None, standardize: bool = False):
    """
    Convert complex batches to scaled or standardized real-valued TF tensors.
    """
    x = complx2real(to_structured(H_in_batch))
    y = complx2real(to_structured(H_perf_batch))

    x = tf.transpose(x, (0, 2, 3, 1))
    y = tf.transpose(y, (0, 2, 3, 1))

    if clip_extrap and pilot_bounds is not None:
        row_min, row_max_idx, col_min, col_max_idx = pilot_bounds
        real = x[:, :, :, 0]
        imag = x[:, :, :, 1]
        
        roi_real = real[:, row_min:row_max_idx, col_min:col_max_idx]
        roi_imag = imag[:, row_min:row_max_idx, col_min:col_max_idx]
        
        min_real = tf.reduce_min(roi_real, axis=(1, 2), keepdims=True)
        max_real = tf.reduce_max(roi_real, axis=(1, 2), keepdims=True)
        min_imag = tf.reduce_min(roi_imag, axis=(1, 2), keepdims=True)
        max_imag = tf.reduce_max(roi_imag, axis=(1, 2), keepdims=True)
        
        real_clipped = tf.clip_by_value(real, min_real, max_real)
        imag_clipped = tf.clip_by_value(imag, min_imag, max_imag)
        
        x = tf.stack([real_clipped, imag_clipped], axis=-1)

    if standardize:
        x_scaled, x_val1, x_val2 = standardizeScaler(x)
        y_scaled, _,    _        = standardizeScaler(y, mean_pre=x_val1, std_pre=x_val2)
    else:
        x_scaled, x_val1, x_val2 = minmaxScaler(x, lower_range=lower_range)
        y_scaled, _,    _        = minmaxScaler(y, min_pre=x_val1, max_pre=x_val2,
                                              lower_range=lower_range)
    return x_scaled, y_scaled, x_val1, x_val2


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def compute_mmse(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    """Mean-squared error between predicted and true complex channels."""
    return float(np.mean(np.abs(H_pred - H_true) ** 2))


def compute_nmse(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    """Normalised MSE (averaged over samples)."""
    num   = np.mean(np.abs(H_pred - H_true) ** 2, axis=(1, 2))
    denom = np.mean(np.abs(H_true) ** 2,           axis=(1, 2))
    return float(np.mean(num / (denom + 1e-30)))


def compute_nmse_db(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    """NMSE in dB."""
    return 10.0 * np.log10(compute_nmse(H_pred, H_true) + 1e-30)


def compute_ssim_batch(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    """
    SSIM computed on the *magnitude* of the complex channel images.
    """
    mag_pred = np.abs(H_pred).astype(np.float32)
    mag_true = np.abs(H_true).astype(np.float32)

    mn    = mag_true.min(axis=(1, 2), keepdims=True)
    mx    = mag_true.max(axis=(1, 2), keepdims=True)
    scale = np.clip(mx - mn, 1e-8, None)
    mag_pred_n = np.clip((mag_pred - mn) / scale, 0.0, 1.0)
    mag_true_n = (mag_true - mn) / scale

    pred_t = tf.constant(mag_pred_n[..., np.newaxis])
    true_t = tf.constant(mag_true_n[..., np.newaxis])
    ssim_vals = tf_ssim(true_t, pred_t, max_val=1.0)
    return float(tf.reduce_mean(ssim_vals).numpy())


# ----------------------------------------------------------------------------
# ONNX Model Export (Architecture + Weights for MATLAB & Python)
# ----------------------------------------------------------------------------
def export_model_to_onnx(model: tf.keras.Model, save_path: str,
                         input_shape=(1, 132, 14, 2)):
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
            model, input_signature=spec, inputs_as_nchw=['input_channel'], output_path=save_path)
        print(f'[ONNX Export] Saved raw ONNX model -> {save_path}')

        # Optionally simplify ONNX graph using onnxsim for MATLAB compatibility
        try:
            from onnxsim import simplify
            import onnx
            model_onnx = onnx.load(save_path)
            model_simp, check = simplify(model_onnx)
            if check:
                simp_path = save_path.replace('.onnx', '_sim.onnx')
                onnx.save(model_simp, simp_path)
                print(f'[ONNX Export] Saved simplified ONNX model (MATLAB compatible) -> {simp_path}')
        except Exception as sim_err:
            print(f'[ONNX Simplification Note] Skipping simplification: {sim_err}')

    except Exception as e:
        print(f'[ONNX Export Warning] Failed to export ONNX model: {e}')



def save_channel_plots_pdf(H_perf_sample: np.ndarray,
                            H_in_sample: np.ndarray,
                            H_pred_sample: np.ndarray,
                            input_type: str, save_dir: str,
                            prefix: str = 'test'):
    """
    Plot and save vector PDF heatmaps of the real part of the channel grids for Sample 1.
    Generates:
      - H_perfect_{prefix}_sample1.pdf
      - H_{input_type}_{prefix}_sample1.pdf
      - H_{input_type}_cnn_{prefix}_sample1.pdf
    """
    try:
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)

        # 1. Perfect Reference Channel PDF
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(H_perf_sample.real, aspect='auto', cmap='viridis')
        fig.colorbar(im, ax=ax)
        ax.set_title(f'Perfect Reference Channel (Real Part) - {prefix.capitalize()} Sample 1', fontsize=14)
        ax.set_xlabel('Symbol Index', fontsize=12)
        ax.set_ylabel('Subcarrier Index', fontsize=12)
        plt.tight_layout()
        pdf_perf = os.path.join(save_dir, f'H_perfect_{prefix}_sample1.pdf')
        plt.savefig(pdf_perf, format='pdf')
        plt.close(fig)

        # 2. Input Channel PDF (H_ls or H_li)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(H_in_sample.real, aspect='auto', cmap='viridis')
        fig.colorbar(im, ax=ax)
        ax.set_title(f'Input Channel H_{input_type} (Real Part) - {prefix.capitalize()} Sample 1', fontsize=14)
        ax.set_xlabel('Symbol Index', fontsize=12)
        ax.set_ylabel('Subcarrier Index', fontsize=12)
        plt.tight_layout()
        pdf_in = os.path.join(save_dir, f'H_{input_type}_{prefix}_sample1.pdf')
        plt.savefig(pdf_in, format='pdf')
        plt.close(fig)

        # 3. CNN Output Channel PDF (H_ls_cnn or H_li_cnn)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(H_pred_sample.real, aspect='auto', cmap='viridis')
        fig.colorbar(im, ax=ax)
        ax.set_title(f'CNN Model Output H_{input_type}_cnn (Real Part) - {prefix.capitalize()} Sample 1', fontsize=14)
        ax.set_xlabel('Symbol Index', fontsize=12)
        ax.set_ylabel('Subcarrier Index', fontsize=12)
        plt.tight_layout()
        pdf_pred = os.path.join(save_dir, f'H_{input_type}_cnn_{prefix}_sample1.pdf')
        plt.savefig(pdf_pred, format='pdf')
        plt.close(fig)

        print(f'[PDF Export] {prefix.capitalize()} channel grid heatmaps saved to: {save_dir}')
    except Exception as e:
        print(f'[PDF Export Warning] Failed to export {prefix} channel heatmaps: {e}')


def save_loss_plot_pdf(history: dict, save_dir: str):
    """
    Plot and save training and validation/evaluation loss curves as separate PDFs.
    Generates:
      - loss_total.pdf (Combined Loss)
      - loss_mse.pdf (MSE component)
      - loss_ssim.pdf (SSIM component)
    """
    try:
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)

        epochs = range(1, len(history['train_loss']) + 1)

        # 1. Total Combined Loss Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs, history['train_loss'], label='Training Total Loss', color='blue', linewidth=2)
        ax.plot(epochs, history['val_loss'], label='Validation Total Loss', color='red', linewidth=2)
        ax.set_title('Total Combined Loss over Epochs', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=12)
        plt.tight_layout()
        pdf_path = os.path.join(save_dir, 'loss_total.pdf')
        plt.savefig(pdf_path, format='pdf')
        plt.close(fig)
        print(f'[PDF Export] Total loss plot saved to: {pdf_path}')

        # 2. MSE Component Loss Plot
        if 'train_mse' in history and 'val_mse' in history:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(epochs, history['train_mse'], label='Training MSE Loss', color='blue', linewidth=2)
            ax.plot(epochs, history['val_mse'], label='Validation MSE Loss', color='red', linewidth=2)
            ax.set_title('MSE Loss Component over Epochs', fontsize=14)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss (MSE)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize=12)
            plt.tight_layout()
            pdf_path = os.path.join(save_dir, 'loss_mse.pdf')
            plt.savefig(pdf_path, format='pdf')
            plt.close(fig)
            print(f'[PDF Export] MSE loss component plot saved to: {pdf_path}')

        # 3. SSIM Component Loss Plot
        if 'train_ssim' in history and 'val_ssim' in history:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(epochs, history['train_ssim'], label='Training SSIM Loss (1-SSIM)', color='blue', linewidth=2)
            ax.plot(epochs, history['val_ssim'], label='Validation SSIM Loss (1-SSIM)', color='red', linewidth=2)
            ax.set_title('SSIM Loss Component (1 - SSIM) over Epochs', fontsize=14)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss (SSIM)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize=12)
            plt.tight_layout()
            pdf_path = os.path.join(save_dir, 'loss_ssim.pdf')
            plt.savefig(pdf_path, format='pdf')
            plt.close(fig)
            print(f'[PDF Export] SSIM loss component plot saved to: {pdf_path}')

    except Exception as e:
        print(f'[PDF Export Warning] Failed to export loss history plots: {e}')


def compute_combined_loss(y_true, y_pred, loss_fn, lower_range, ssim_weight, standardize=False):
    """
    Computes a combined loss of MSE and SSIM:
      total_loss = (1 - ssim_weight) * MSE + ssim_weight * (1 - SSIM)
    """
    # 1. MSE Loss
    mse_val = loss_fn(y_true, y_pred)
    
    # 2. SSIM Loss
    if standardize:
        max_val = tf.reduce_max(y_true) - tf.reduce_min(y_true)
        max_val = tf.maximum(max_val, 1e-8)
    else:
        max_val = tf.cast(2.0 if lower_range == -1 else 1.0, tf.float32)
        
    ssim_val = tf.image.ssim(y_true, y_pred, max_val=max_val)
    ssim_loss = tf.reduce_mean(1.0 - ssim_val)
    
    # Combine
    combined = (1.0 - ssim_weight) * mse_val + ssim_weight * ssim_loss
    return combined, mse_val, ssim_loss


# ----------------------------------------------------------------------------
# Training step (compiled for speed)
# ----------------------------------------------------------------------------
@tf.function
def _train_step(model, x_scaled, y_scaled, optimizer, loss_fn, lower_range, ssim_weight, standardize):
    with tf.GradientTape() as tape:
        residual, _  = model(x_scaled, training=True)
        x_corrected  = x_scaled + residual
        
        # Compute combined loss components
        combined_l, mse_l, ssim_l = compute_combined_loss(y_scaled, x_corrected, loss_fn, lower_range, ssim_weight, standardize)
        
        reg_loss     = 0.001 * tf.reduce_mean(tf.square(residual))
        total_loss   = combined_l + reg_loss
        if model.losses:
            total_loss += tf.add_n(model.losses)
            
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, mse_l, ssim_l


# ----------------------------------------------------------------------------
# Inference: produce corrected complex channel
# ----------------------------------------------------------------------------
def infer_channel(model: DnCNN_ResNet_CrossAttention,
                  H_perf: np.ndarray, H_in: np.ndarray,
                  batch_size: int, lower_range: int,
                  clip_extrap: bool = False,
                  pilot_bounds: tuple = None,
                  standardize: bool = False) -> np.ndarray:
    """
    Run the trained model on a full dataset and return the predicted
    complex channel of shape [N, n_subc, n_symb].
    """
    N     = H_perf.shape[0]
    steps = N // batch_size
    out   = []

    for i in range(steps):
        sl   = slice(i * batch_size, (i + 1) * batch_size)
        x_sc, _, x_val1, x_val2 = preprocess_batch(H_perf[sl], H_in[sl], lower_range,
                                                   clip_extrap=clip_extrap, pilot_bounds=pilot_bounds,
                                                   standardize=standardize)
        residual, _ = model(x_sc, training=False)
        x_corr      = x_sc + residual
        if standardize:
            x_denorm = deStandardize(x_corr, x_val1, x_val2)
        else:
            x_denorm = deMinMax(x_corr, x_val1, x_val2, lower_range=lower_range)
        x_np        = x_denorm.numpy()
        out.append(x_np[..., 0] + 1j * x_np[..., 1])

    # Remainder batch
    if N % batch_size:
        sl   = slice(steps * batch_size, N)
        x_sc, _, x_val1, x_val2 = preprocess_batch(H_perf[sl], H_in[sl], lower_range,
                                                   clip_extrap=clip_extrap, pilot_bounds=pilot_bounds,
                                                   standardize=standardize)
        residual, _ = model(x_sc, training=False)
        x_corr      = x_sc + residual
        if standardize:
            x_denorm = deStandardize(x_corr, x_val1, x_val2)
        else:
            x_denorm = deMinMax(x_corr, x_val1, x_val2, lower_range=lower_range)
        x_np        = x_denorm.numpy()
        out.append(x_np[..., 0] + 1j * x_np[..., 1])

    return np.concatenate(out, axis=0)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    # -- Argument parsing ----------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Single-dataset CNN channel estimator (MSE + SSIM training loss decay schedule).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--snr', type=int, default=DEFAULT_SNR,
                        choices=sorted(SNR_FOLDER_MAP),
                        help='Channel SNR in dB')
    parser.add_argument('--input-type', type=str, default=DEFAULT_INPUT_TYPE,
                        choices=['prac', 'li', 'li_ori', 'ls', 'ls_ori'],
                        help='Noisy estimate used as CNN input (prac, li, li_ori, ls, ls_ori)')
    parser.add_argument('--epochs', type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help='Adam learning rate')
    parser.add_argument('--train-frac', type=float, default=DEFAULT_TRAIN_FRAC,
                        help='Fraction of data for training (0 < x < 1)')
    parser.add_argument('--val-frac', type=float, default=DEFAULT_VAL_FRAC,
                        help='Fraction of data for validation (remainder = test)')
    parser.add_argument('--n-blocks', type=int, default=DEFAULT_N_BLOCKS,
                        help='Number of residual blocks in CNNGenerator')
    parser.add_argument('--save-model', action='store_true',
                        default=DEFAULT_SAVE_MODEL,
                        help='Save trained model weights to disk')
    parser.add_argument('--save-dir', type=str, default=DEFAULT_SAVE_DIR,
                        help='Directory to save model.')
    parser.add_argument('--data-root', type=str, default=DEFAULT_DATA_ROOT,
                        help='Root folder for OpenNTN channel data')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU even if available')
    parser.add_argument('--test-code', action='store_true',
                        help='Quick smoke-test with small dataset')
    parser.add_argument('--clip-extrap', action='store_true',
                        default=DEFAULT_CLIP_EXTRAP,
                        help='Clip extrapolation values of the input grid to the pilot region bounds')
    parser.add_argument('--ssim-weight-start', type=float, default=DEFAULT_SSIM_START,
                        help='Initial importance weight for SSIM loss at epoch 0.')
    parser.add_argument('--ssim-weight-end', type=float, default=DEFAULT_SSIM_END,
                        help='Final importance weight for SSIM loss at the last epoch.')
    parser.add_argument('--standardize', action='store_true',
                        help='Use sample-wise standardization (mean/var) instead of min-max scaling')

    args = parser.parse_args()

    # -- Validate ------------------------------------------------------------
    if args.train_frac + args.val_frac >= 1.0:
        parser.error('--train-frac + --val-frac must be < 1.0 (remainder is test).')
    if not (0.0 <= args.ssim_weight_start <= 1.0) or not (0.0 <= args.ssim_weight_end <= 1.0):
        parser.error('Both ssim-weight-start and ssim-weight-end must be between 0.0 and 1.0.')

    # -- GPU setup ------------------------------------------------------------
    if args.no_gpu:
        tf.config.set_visible_devices([], 'GPU')
        print('[GPU] Disabled by --no-gpu flag.')
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f'[GPU] Using: {[g.name for g in gpus]}')
        else:
            print('[GPU] None found – running on CPU.')

    # -- Config banner --------------------------------------------------------
    test_frac = 1.0 - args.train_frac - args.val_frac
    print('\n' + '=' * 58)
    print('  Single-Dataset CNN (Decaying MSE + SSIM Combination Loss)')
    print('=' * 58)
    print(f'  SNR               : {args.snr:+d} dB')
    print(f'  Input type        : {args.input_type}')
    print(f'  Epochs            : {args.epochs}')
    print(f'  Batch size        : {args.batch_size}')
    print(f'  Learning rate     : {args.lr}')
    print(f'  SSIM Weight Start : {args.ssim_weight_start:.3f}')
    print(f'  SSIM Weight End   : {args.ssim_weight_end:.3f}')
    print(f'  Split             : {args.train_frac:.0%} / {args.val_frac:.0%} '
          f'/ {test_frac:.0%}  (train / val / test)')
    print(f'  CNN n_blocks      : {args.n_blocks}')
    print(f'  Clip extrapolation: {args.clip_extrap}')
    print('=' * 58 + '\n')

    # -- Data loading & splitting ---------------------------------------------
    mat_path = get_data_path(args.data_root, args.snr)
    print(f'[Data] {mat_path}')
    H_perfect, H_input, mat_dict, H_perfect_ori, H_li_benchmark_grid = load_mat_data(mat_path, args.input_type)
    N = H_perfect.shape[0]

    # Calculate pilot bounds dynamically
    pilot_cols = mat_dict['pilot_cols'].squeeze() - 1
    pilot_rows = mat_dict['pilot_rows'].squeeze() - 1
    row_min = int(np.min(pilot_rows))
    row_max = int(np.max(pilot_rows))
    col_min = int(np.min(pilot_cols))
    col_max = int(np.max(pilot_cols))
    pilot_bounds = (row_min, row_max + 1, col_min, col_max + 1)

    idx_train, idx_val, idx_test = split_indices(
        N, args.train_frac, args.val_frac, seed=1234)

    # -- Test-code mode: shrink to a tiny subset -----------------------------
    if args.test_code:
        print('[test-code] Limiting dataset to a tiny subset for quick testing.')
        idx_train = idx_train[:TEST_CODE_N_TRAIN]
        idx_val   = idx_val  [:TEST_CODE_N_VAL]
        idx_test  = idx_test [:TEST_CODE_N_TEST]
        args.epochs = TEST_CODE_EPOCHS
        print(f'[test-code] Epochs overridden to {args.epochs}')

    print(f'[Data] N={N}  '
          f'train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}')

    # -- Model, optimiser, loss -----------------------------------------------
    model     = DnCNN_ResNet_CrossAttention(n_blocks=args.n_blocks, pilot_rows=pilot_rows, pilot_cols=pilot_cols)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr,
                                         beta_1=0.5, beta_2=0.9)
    loss_fn   = tf.keras.losses.MeanSquaredError()

    # -- Save directory -------------------------------------------------------
    if args.save_dir:
        save_dir = os.path.abspath(args.save_dir)
    else:
        ssim_start_str = str(args.ssim_weight_start).replace('.', '_')
        ssim_end_str = str(args.ssim_weight_end).replace('.', '_')
        suffix = 'standardize' if args.standardize else 'ssim_decay'
        save_dir = os.path.join(THIS_DIR, 'trained_models',
                                f'SNR_{args.snr}dB_{args.input_type}_{suffix}_s{ssim_start_str}_e{ssim_end_str}')
    os.makedirs(save_dir, exist_ok=True)
    print(f'[Save] Model dir : {save_dir}\n')

    # -- Training --------------------------------------------------------------
    lower_range     = DEFAULT_LOWER_RANGE
    n_train_batches = len(idx_train) // args.batch_size
    n_val_batches   = len(idx_val)   // args.batch_size

    best_val_loss = float('inf')
    best_epoch    = 0
    history = {
        'train_loss': [],
        'train_mse':  [],
        'train_ssim': [],
        'val_loss':   [],
        'val_mse':    [],
        'val_ssim':   [],
        'val_nmse':   [],
        'ssim_weight_history': [],
    }

    print(f'[Train] {args.epochs} epochs  |  {n_train_batches} batches/epoch\n')
    t_start = time.perf_counter()

    # Convert standardize flag to TF constant
    standardize_tf = tf.constant(args.standardize, dtype=tf.bool)

    for epoch in range(args.epochs):
        idx_e = np.random.default_rng(epoch).permutation(idx_train)

        # Calculate dynamic SSIM weight for the current epoch (linear decay schedule)
        if args.epochs > 1:
            epoch_ssim_weight = args.ssim_weight_start + (epoch / (args.epochs - 1)) * (args.ssim_weight_end - args.ssim_weight_start)
        else:
            epoch_ssim_weight = args.ssim_weight_start
            
        history['ssim_weight_history'].append(epoch_ssim_weight)
        
        # Convert to TensorFlow constant to avoid retracing warnings
        ssim_weight_tf = tf.constant(epoch_ssim_weight, dtype=tf.float32)

        # ---------- Training pass ----------
        ep_train_loss = 0.0
        ep_train_mse  = 0.0
        ep_train_ssim = 0.0
        for b in range(n_train_batches):
            batch_idx = idx_e[b * args.batch_size:(b + 1) * args.batch_size]
            h_p = H_perfect[batch_idx]
            h_i = H_input[batch_idx]
            x_sc, y_sc, _, _ = preprocess_batch(h_p, h_i, lower_range, clip_extrap=args.clip_extrap, pilot_bounds=pilot_bounds, standardize=args.standardize)
            
            total_l, mse_l, ssim_l = _train_step(model, x_sc, y_sc, optimizer, loss_fn, lower_range, ssim_weight_tf, standardize_tf)
            ep_train_loss += total_l.numpy()
            ep_train_mse  += mse_l.numpy()
            ep_train_ssim += ssim_l.numpy()

        avg_train_loss = ep_train_loss / max(n_train_batches, 1)
        avg_train_mse  = ep_train_mse / max(n_train_batches, 1)
        avg_train_ssim = ep_train_ssim / max(n_train_batches, 1)

        # ---------- Validation pass ----------
        ep_val_loss = 0.0
        ep_val_mse  = 0.0
        ep_val_ssim = 0.0
        ep_val_nmse = 0.0
        for b in range(n_val_batches):
            batch_idx = idx_val[b * args.batch_size:(b + 1) * args.batch_size]
            h_p = H_perfect[batch_idx]
            h_i = H_input[batch_idx]
            x_sc, y_sc, x_val1, x_val2 = preprocess_batch(h_p, h_i, lower_range, clip_extrap=args.clip_extrap, pilot_bounds=pilot_bounds, standardize=args.standardize)
            residual, _ = model(x_sc, training=False)
            x_corr      = x_sc + residual
            
            # Use current epoch's SSIM weight for validation loss to match training
            comb_l, mse_l, ssim_l = compute_combined_loss(y_sc, x_corr, loss_fn, lower_range, ssim_weight_tf, standardize=args.standardize)
            ep_val_loss += comb_l.numpy()
            ep_val_mse  += mse_l.numpy()
            ep_val_ssim += ssim_l.numpy()

            # Quick NMSE in scaled domain
            x_np = x_corr.numpy()
            y_np = y_sc.numpy()
            diff_sq = np.mean((x_np - y_np) ** 2, axis=(1, 2, 3))
            ref_sq  = np.mean(y_np ** 2,           axis=(1, 2, 3))
            ep_val_nmse += float(np.mean(diff_sq / (ref_sq + 1e-30)))

        avg_val_loss = ep_val_loss / max(n_val_batches, 1)
        avg_val_mse  = ep_val_mse / max(n_val_batches, 1)
        avg_val_ssim = ep_val_ssim / max(n_val_batches, 1)
        avg_val_nmse = ep_val_nmse / max(n_val_batches, 1)

        history['train_loss'].append(avg_train_loss)
        history['train_mse'].append(avg_train_mse)
        history['train_ssim'].append(avg_train_ssim)
        history['val_loss'].append(avg_val_loss)
        history['val_mse'].append(avg_val_mse)
        history['val_ssim'].append(avg_val_ssim)
        history['val_nmse'].append(avg_val_nmse)

        # -- Print every 10 epochs (and first / last) --
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            elapsed = time.perf_counter() - t_start
            print(f'Epoch [{epoch+1:>4d}/{args.epochs}]  '
                  f'TrainLoss={avg_train_loss:.6f} (MSE={avg_train_mse:.6f}, SSIM_W={epoch_ssim_weight:.3f})  '
                  f'ValLoss={avg_val_loss:.6f} (MSE={avg_val_mse:.6f}, SSIM_loss={avg_val_ssim:.4f})  '
                  f'ValNMSE={avg_val_nmse:.6f}  '
                  f't={elapsed:.1f}s')

        # -- Save best model --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch    = epoch + 1
            if args.save_model:
                export_model_to_onnx(model, os.path.join(save_dir, 'best_net.onnx'))
                with open(os.path.join(save_dir, 'best_epoch.txt'), 'w') as fh:
                    fh.write(f'best_epoch       = {best_epoch}\n'
                             f'best_val_loss    = {best_val_loss:.8f}\n'
                             f'snr              = {args.snr} dB\n'
                             f'input_type       = {args.input_type}\n'
                             f'standardize      = {args.standardize}\n'
                             f'total_epochs     = {args.epochs}\n'
                             f'batch_size       = {args.batch_size}\n'
                             f'learning_rate    = {args.lr}\n'
                             f'ssim_weight_best = {epoch_ssim_weight:.6f}\n'
                             f'mse_weight_best  = {1.0 - epoch_ssim_weight:.6f}\n'
                             f'ssim_weight_start= {args.ssim_weight_start}\n'
                             f'ssim_weight_end  = {args.ssim_weight_end}\n'
                             f'n_blocks         = {args.n_blocks}\n'
                             f'total_samples    = {N}\n'
                             f'n_train_samples  = {len(idx_train)}\n'
                             f'n_val_samples    = {len(idx_val)}\n'
                             f'n_test_samples   = {len(idx_test)}\n'
                             f'test_code_mode   = {args.test_code}\n'
                             f'data_path        = {mat_path}\n')

    if args.save_model:
        export_model_to_onnx(model, os.path.join(save_dir, 'final_net.onnx'))
        print(f'\n[Save] Final model -> {os.path.join(save_dir, "final_net.onnx")}')
        print(f'[Save] Best  model -> {os.path.join(save_dir, "best_net.onnx")}'
              f'  (epoch {best_epoch})')

    # Save training history (.mat)
    hist_path = os.path.join(save_dir, 'training_history.mat')
    scipy.io.savemat(hist_path, {
        'train_loss': np.array(history['train_loss']),
        'train_mse':  np.array(history['train_mse']),
        'train_ssim': np.array(history['train_ssim']),
        'val_loss':   np.array(history['val_loss']),
        'val_mse':    np.array(history['val_mse']),
        'val_ssim':   np.array(history['val_ssim']),
        'val_nmse':   np.array(history['val_nmse']),
        'ssim_weight_history': np.array(history['ssim_weight_history']),
        'snr':        args.snr,
        'input_type': args.input_type,
        'standardize': args.standardize,
        'n_epochs':   args.epochs,
        'best_epoch': best_epoch,
        'ssim_weight_start': args.ssim_weight_start,
        'ssim_weight_end':   args.ssim_weight_end,
    })
    print(f'[Save] Training history -> {hist_path}')

    # Save training loss curves plot PDF
    save_loss_plot_pdf(history, save_dir)

    # -- Evaluation on VALIDATION set -----------------------------------------
    print('\n' + '-' * 58)
    print('[Eval] Validation set ...')
    H_pred_val = infer_channel(model,
                               H_perfect[idx_val], H_input[idx_val],
                               args.batch_size, lower_range,
                               clip_extrap=args.clip_extrap,
                               pilot_bounds=pilot_bounds,
                               standardize=args.standardize)

    mmse_val = compute_mmse(H_pred_val, H_perfect[idx_val])
    nmse_val = compute_nmse(H_pred_val, H_perfect[idx_val])
    ssim_val = compute_ssim_batch(H_pred_val, H_perfect[idx_val])
    nmse_val_db = 10.0 * np.log10(nmse_val + 1e-30)

    # Raw noisy baseline on validation
    mmse_in_val = compute_mmse(H_input[idx_val], H_perfect[idx_val])
    nmse_in_val = compute_nmse(H_input[idx_val], H_perfect[idx_val])
    ssim_in_val = compute_ssim_batch(H_input[idx_val], H_perfect[idx_val])
    nmse_in_val_db = 10.0 * np.log10(nmse_in_val + 1e-30)

    # -- Evaluation on TEST set ------------------------------------------------
    print('[Eval] Test set ...')
    H_pred_test = infer_channel(model,
                                H_perfect[idx_test], H_input[idx_test],
                                args.batch_size, lower_range,
                                clip_extrap=args.clip_extrap,
                                pilot_bounds=pilot_bounds,
                                standardize=args.standardize)

    mmse_test = compute_mmse(H_pred_test, H_perfect[idx_test])
    nmse_test = compute_nmse(H_pred_test, H_perfect[idx_test])
    ssim_test = compute_ssim_batch(H_pred_test, H_perfect[idx_test])
    nmse_test_db = 10.0 * np.log10(nmse_test + 1e-30)

    # Raw noisy baseline on test
    mmse_in_test = compute_mmse(H_input[idx_test], H_perfect[idx_test])
    nmse_in_test = compute_nmse(H_input[idx_test], H_perfect[idx_test])
    ssim_in_test = compute_ssim_batch(H_input[idx_test], H_perfect[idx_test])
    nmse_in_test_db = 10.0 * np.log10(nmse_in_test + 1e-30)

    # -- Evaluation on TRAINING set -------------------------------------------
    print('[Eval] Training set ...')
    H_pred_train = infer_channel(model,
                                 H_perfect[idx_train], H_input[idx_train],
                                 args.batch_size, lower_range,
                                 clip_extrap=args.clip_extrap,
                                 pilot_bounds=pilot_bounds,
                                 standardize=args.standardize)

    mmse_train = compute_mmse(H_pred_train, H_perfect[idx_train])
    nmse_train = compute_nmse(H_pred_train, H_perfect[idx_train])
    ssim_train = compute_ssim_batch(H_pred_train, H_perfect[idx_train])
    nmse_train_db = 10.0 * np.log10(nmse_train + 1e-30)

    # Raw noisy baseline on training
    mmse_in_train = compute_mmse(H_input[idx_train], H_perfect[idx_train])
    nmse_in_train = compute_nmse(H_input[idx_train], H_perfect[idx_train])
    ssim_in_train = compute_ssim_batch(H_input[idx_train], H_perfect[idx_train])
    nmse_in_train_db = 10.0 * np.log10(nmse_in_train + 1e-30)

    # -- LMMSE Baseline Estimation ---------------------------------------------
    print('\n' + '-' * 58)
    print('[LMMSE] Calculating LMMSE baseline estimated channel...')
    has_lmmse = False
    try:
        ls_key = None
        for k in ['H_ls_pilots', 'H_ls_pilots_ori', 'H_LS_comp', 'H_LS_full']:
            if k in mat_dict and mat_dict[k].size > 0:
                ls_key = k
                break
                
        if ls_key is None or 'pilot_rows' not in mat_dict or 'pilot_cols' not in mat_dict:
            raise KeyError("Required pilot keys not found in mat_dict for LMMSE.")
            
        H_ls_pilots = mat_dict[ls_key]
        if H_ls_pilots.shape[0] != N and H_ls_pilots.shape[1] == N:
            H_ls_pilots = H_ls_pilots.T
            
        pilot_rows = np.squeeze(mat_dict['pilot_rows']).astype(int) - 1
        pilot_cols = np.squeeze(mat_dict['pilot_cols']).astype(int) - 1
        
        H_perfect_pilots = H_perfect[:, pilot_rows, pilot_cols]
        H_perfect_vec = H_perfect.reshape(N, -1)
        
        # Complex covariance/correlation matrices
        R_HP = np.matmul(H_perfect_vec.transpose(1, 0).conj(), H_perfect_pilots) / N
        R_PP = np.matmul(H_perfect_pilots.transpose(1, 0).conj(), H_perfect_pilots) / N
        
        noise_var = float(np.mean(np.abs(H_ls_pilots - H_perfect_pilots) ** 2))
        
        C = R_PP + noise_var * np.eye(R_PP.shape[0], dtype=np.complex128)
        inv_C = np.linalg.inv(C)
        W = np.matmul(R_HP, inv_C)
        
        H_lmmse_train = np.matmul(H_ls_pilots[idx_train], W.T).reshape(-1, 132, 14)
        H_lmmse_val = np.matmul(H_ls_pilots[idx_val], W.T).reshape(-1, 132, 14)
        H_lmmse_test = np.matmul(H_ls_pilots[idx_test], W.T).reshape(-1, 132, 14)
        
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

    # -- Results table ---------------------------------------------------------
    hdr = f'  SNR={args.snr:+d} dB  |  input type: {args.input_type}  |  ssim_decay: {args.ssim_weight_start:.3f} -> {args.ssim_weight_end:.3f}'
    print('\n' + '=' * 80)
    print('  EVALUATION RESULTS (SSIM Weight Decay)')
    print('=' * 80)
    print(hdr)
    print('-' * 80)
    if has_lmmse:
        print(f'  {"Metric":<14} {"Raw "+args.input_type:<18} {"CNN output":<18} {"LMMSE Baseline":<18}')
        print('-' * 80)
        print(f'  {"[TRAIN]":<14}')
        print(f'  {"  MMSE":<14} {mmse_in_train:<18.6e} {mmse_train:<18.6e} {mmse_lmmse_train:<18.6e}')
        print(f'  {"  NMSE":<14} {nmse_in_train:<18.6f} {nmse_train:<18.6f} {nmse_lmmse_train:<18.6f}')
        print(f'  {"  NMSE(dB)":<14} {nmse_in_train_db:<18.2f} {nmse_train_db:<18.2f} {nmse_db_lmmse_train:<18.2f}')
        print(f'  {"  SSIM":<14} {ssim_in_train:<18.6f} {ssim_train:<18.6f} {ssim_lmmse_train:<18.6f}')
        print('-' * 80)
        print(f'  {"[VAL]":<14}')
        print(f'  {"  MMSE":<14} {mmse_in_val:<18.6e} {mmse_val:<18.6e} {mmse_lmmse_val:<18.6e}')
        print(f'  {"  NMSE":<14} {nmse_in_val:<18.6f} {nmse_val:<18.6f} {nmse_lmmse_val:<18.6f}')
        print(f'  {"  NMSE(dB)":<14} {nmse_in_val_db:<18.2f} {nmse_val_db:<18.2f} {nmse_db_lmmse_val:<18.2f}')
        print(f'  {"  SSIM":<14} {ssim_in_val:<18.6f} {ssim_val:<18.6f} {ssim_lmmse_val:<18.6f}')
        print('-' * 80)
        print(f'  {"[TEST]":<14}')
        print(f'  {"  MMSE":<14} {mmse_in_test:<18.6e} {mmse_test:<18.6e} {mmse_lmmse_test:<18.6e}')
        print(f'  {"  NMSE":<14} {nmse_in_test:<18.6f} {nmse_test:<18.6f} {nmse_lmmse_test:<18.6f}')
        print(f'  {"  NMSE(dB)":<14} {nmse_in_test_db:<18.2f} {nmse_test_db:<18.2f} {nmse_db_lmmse_test:<18.2f}')
        print(f'  {"  SSIM":<14} {ssim_in_test:<18.6f} {ssim_test:<18.6f} {ssim_lmmse_test:<18.6f}')
    else:
        print(f'  {"Metric":<14} {"Raw "+args.input_type:<20} {"CNN output":<20}')
        print('-' * 60)
        print(f'  {"[TRAIN]":<14}')
        print(f'  {"  MMSE":<14} {mmse_in_train:<20.6e} {mmse_train:<20.6e}')
        print(f'  {"  NMSE":<14} {nmse_in_train:<20.6f} {nmse_train:<20.6f}')
        print(f'  {"  NMSE(dB)":<14} {nmse_in_train_db:<20.2f} {nmse_train_db:<20.2f}')
        print(f'  {"  SSIM":<14} {ssim_in_train:<20.6f} {ssim_train:<20.6f}')
        print('-' * 60)
        print(f'  {"[VAL]":<14}')
        print(f'  {"  MMSE":<14} {mmse_in_val:<20.6e} {mmse_val:<20.6e}')
        print(f'  {"  NMSE":<14} {nmse_in_val:<20.6f} {nmse_val:<20.6f}')
        print(f'  {"  NMSE(dB)":<14} {nmse_in_val_db:<20.2f} {nmse_val_db:<20.2f}')
        print(f'  {"  SSIM":<14} {ssim_in_val:<20.6f} {ssim_val:<20.6f}')
        print('-' * 60)
        print(f'  {"[TEST]":<14}')
        print(f'  {"  MMSE":<14} {mmse_in_test:<20.6e} {mmse_test:<20.6e}')
        print(f'  {"  NMSE":<14} {nmse_in_test:<20.6f} {nmse_test:<20.6f}')
        print(f'  {"  NMSE(dB)":<14} {nmse_in_test_db:<20.2f} {nmse_test_db:<20.2f}')
        print(f'  {"  SSIM":<14} {ssim_in_test:<20.6f} {ssim_test:<20.6f}')
    print('=' * 80)

    # -- Save evaluation .mat --------------------------------------------------
    eval_path = os.path.join(save_dir, 'evaluation_results.mat')
    eval_dict = {
        # --- Train ---
        'mmse_train':      mmse_train,
        'nmse_train':      nmse_train,
        'nmse_train_db':   nmse_train_db,
        'ssim_train':      ssim_train,
        'mmse_input_train': mmse_in_train,
        'nmse_input_train': nmse_in_train,
        'nmse_input_train_db': nmse_in_train_db,
        'ssim_input_train': ssim_in_train,
        # --- Validation ---
        'mmse_val':        mmse_val,
        'nmse_val':        nmse_val,
        'nmse_val_db':     nmse_val_db,
        'ssim_val':        ssim_val,
        'mmse_input_val':  mmse_in_val,
        'nmse_input_val':  nmse_in_val,
        'nmse_input_val_db': nmse_in_val_db,
        'ssim_input_val':  ssim_in_val,
        # --- Test ---
        'mmse_test':       mmse_test,
        'nmse_test':       nmse_test,
        'nmse_test_db':    nmse_test_db,
        'ssim_test':       ssim_test,
        'mmse_input_test': mmse_in_test,
        'nmse_input_test': nmse_in_test,
        'nmse_input_test_db': nmse_in_test_db,
        'ssim_input_test': ssim_in_test,
        # --- Meta ---
        'snr':             args.snr,
        'input_type':      args.input_type,
        'n_train':         len(idx_train),
        'n_val':           len(idx_val),
        'n_test':          len(idx_test),
        'best_epoch':      best_epoch,
        'ssim_weight_start': args.ssim_weight_start,
        'ssim_weight_end':   args.ssim_weight_end,
    }

    if has_lmmse:
        eval_dict.update({
            # --- LMMSE ---
            'mmse_lmmse_train':      mmse_lmmse_train,
            'nmse_lmmse_train':      nmse_lmmse_train,
            'nmse_db_lmmse_train':   nmse_db_lmmse_train,
            'ssim_lmmse_train':      ssim_lmmse_train,
            
            'mmse_lmmse_val':        mmse_lmmse_val,
            'nmse_lmmse_val':        nmse_lmmse_val,
            'nmse_db_lmmse_val':     nmse_db_lmmse_val,
            'ssim_lmmse_val':        ssim_lmmse_val,
            
            'mmse_lmmse_test':       mmse_lmmse_test,
            'nmse_lmmse_test':       nmse_lmmse_test,
            'nmse_db_lmmse_test':    nmse_db_lmmse_test,
            'ssim_lmmse_test':       ssim_lmmse_test,
            
        })

    scipy.io.savemat(eval_path, eval_dict)
    print(f'[Save] Evaluation results -> {eval_path}')

    # Save the test set channel grids for BER / metrics visualization
    test_grids_path = os.path.join(save_dir, 'testChannel.mat')
    # Resolve H_LS_test: retrieve H_ls_pilots sequence
    H_ls_pilots = mat_dict.get('H_ls_pilots', mat_dict.get('H_ls_pilots_ori'))
    if H_ls_pilots is not None:
        if H_ls_pilots.ndim == 2:
            if H_ls_pilots.shape[0] != H_perfect.shape[0] and H_ls_pilots.shape[1] == H_perfect.shape[0]:
                H_ls_pilots = H_ls_pilots.T
            H_LS_test_val = H_ls_pilots[idx_test]
        else:
            H_LS_test_val = H_input[idx_test]
    else:
        H_LS_test_val = H_input[idx_test]

    test_grids_dict = {
        'H_original_test': H_perfect_ori[idx_test],      # Original channel before Doppler comp
        'H_perfect_test': H_perfect[idx_test],            # Effective channel after Doppler comp
        'H_LS_test': H_LS_test_val,                       # LS pilots sequence
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
        if md_matches:
            shutil.copy(md_matches[0], save_dir)
            print(f"[Save] Copied dataset readme to: {save_dir}")
    except Exception as e:
        print(f"[Save Warning] Failed to copy metadata readme: {e}")

    # -- Export PDF Heatmap Visualizations & Save Complex Grids for Test and Train Sample 1 ----
    if len(idx_test) > 0:
        sample_idx = idx_test[0]
        sample_mat_path = os.path.join(save_dir, 'channel_grids_test_sample1.mat')
        sample_dict = {
            'H_perfect': H_perfect[sample_idx],
            f'H_{args.input_type}': H_input[sample_idx],
            f'H_{args.input_type}_cnn': H_pred_test[0],
            'snr': args.snr,
            'input_type': args.input_type
        }
        if has_lmmse:
            sample_dict[f'H_{args.input_type}_lmmse'] = H_lmmse_test[0]
        scipy.io.savemat(sample_mat_path, sample_dict)
        print(f'[Save] Complex channel grids (Test Sample 1) -> {sample_mat_path}')

        save_channel_plots_pdf(H_perfect[sample_idx],
                               H_input[sample_idx],
                               H_pred_test[0],
                               args.input_type, save_dir,
                               prefix='test')

    if len(idx_train) > 0:
        train_sample_idx = idx_train[0]
        # Run inference on the first training sample
        H_pred_train_sample = infer_channel(
            model,
            H_perfect[train_sample_idx:train_sample_idx+1],
            H_input[train_sample_idx:train_sample_idx+1],
            batch_size=1,
            lower_range=lower_range,
            clip_extrap=args.clip_extrap,
            pilot_bounds=pilot_bounds
        )[0]

        train_sample_mat_path = os.path.join(save_dir, 'channel_grids_train_sample1.mat')
        train_sample_dict = {
            'H_perfect': H_perfect[train_sample_idx],
            f'H_{args.input_type}': H_input[train_sample_idx],
            f'H_{args.input_type}_cnn': H_pred_train_sample,
            'snr': args.snr,
            'input_type': args.input_type
        }
        if has_lmmse:
            train_sample_dict[f'H_{args.input_type}_lmmse'] = H_lmmse_train[0]
        scipy.io.savemat(train_sample_mat_path, train_sample_dict)
        print(f'[Save] Complex channel grids (Train Sample 1) -> {train_sample_mat_path}')

        save_channel_plots_pdf(H_perfect[train_sample_idx],
                               H_input[train_sample_idx],
                               H_pred_train_sample,
                               args.input_type, save_dir,
                               prefix='train')

    # Write final_epoch.txt report
    report_path = os.path.join(save_dir, 'final_epoch.txt')
    try:
        with open(report_path, 'w') as fh:
            fh.write(
                "========================================================================\n"
                "FINAL EVALUATION METRICS COMPARISON\n"
                "========================================================================\n"
                f"SNR: {args.snr} dB | Input Type: {args.input_type} | SSIM Weight: {args.ssim_weight_start:.3f} -> {args.ssim_weight_end:.3f}\n\n"
            )
            
            def write_set_block(set_name, raw_mmse, raw_nmse, raw_nmse_db, raw_ssim,
                                cnn_mmse, cnn_nmse, cnn_nmse_db, ssim_cnn,
                                lmmse_mmse=None, lmmse_nmse=None, lmmse_nmse_db=None, lmmse_ssim=None):
                fh.write(f"========================================================================\n")
                fh.write(f"FINAL EVALUATION METRICS COMPARISON ({set_name} SET)\n")
                fh.write(f"========================================================================\n")
                if lmmse_mmse is not None:
                    fh.write(f"  {'Metric':<14} {'Raw Input':<16} {'CNN Output':<16} {'LMMSE Baseline':<16} {'Improvement?'}\n")
                    fh.write(f"  {'-'*12:<14} {'-'*14:<16} {'-'*14:<16} {'-'*14:<16} {'-'*12}\n")
                    fh.write(f"  {'MMSE (MSE)':<14} {raw_mmse:<16.6e} {cnn_mmse:<16.6e} {lmmse_mmse:<16.6e} {'Yes' if cnn_mmse < raw_mmse else 'No'}\n")
                    fh.write(f"  {'NMSE':<14} {raw_nmse:<16.6f} {cnn_nmse:<16.6f} {lmmse_nmse:<16.6f} {'Yes' if cnn_mmse < raw_mmse else 'No'}\n")
                    fh.write(f"  {'NMSE (dB)':<14} {raw_nmse_db:<16.2f} {cnn_nmse_db:<16.2f} {lmmse_nmse_db:<16.2f} {'Yes' if cnn_nmse_db < raw_nmse_db else 'No'}\n")
                    fh.write(f"  {'SSIM':<14} {raw_ssim:<16.6f} {ssim_cnn:<16.6f} {lmmse_ssim:<16.6f} {'Yes' if ssim_cnn > raw_ssim else 'No'}\n")
                else:
                    fh.write(f"  {'Metric':<14} {'Raw Input':<20} {'CNN Output':<20} {'Improvement?'}\n")
                    fh.write(f"  {'-'*12:<14} {'-'*18:<20} {'-'*18:<20} {'-'*12}\n")
                    fh.write(f"  {'MMSE (MSE)':<14} {raw_mmse:<20.6e} {cnn_mmse:<20.6e} {'Yes' if cnn_mmse < raw_mmse else 'No'}\n")
                    fh.write(f"  {'NMSE':<14} {raw_nmse:<20.6f} {cnn_nmse:<20.6f} {'Yes' if cnn_mmse < raw_mmse else 'No'}\n")
                    fh.write(f"  {'NMSE (dB)':<14} {raw_nmse_db:<20.2f} {cnn_nmse_db:<20.2f} {'Yes' if cnn_nmse_db < raw_nmse_db else 'No'}\n")
                    fh.write(f"  {'SSIM':<14} {raw_ssim:<20.6f} {ssim_cnn:<20.6f} {'Yes' if ssim_cnn > raw_ssim else 'No'}\n")
                fh.write("========================================================================\n\n")

            if has_lmmse:
                write_set_block("TEST", mmse_in_test, nmse_in_test, nmse_in_test_db, ssim_in_test,
                                mmse_test, nmse_test, nmse_test_db, ssim_test,
                                mmse_lmmse_test, nmse_lmmse_test, nmse_db_lmmse_test, ssim_lmmse_test)
                write_set_block("VALIDATION", mmse_in_val, nmse_in_val, nmse_in_val_db, ssim_in_val,
                                mmse_val, nmse_val, nmse_val_db, ssim_val,
                                mmse_lmmse_val, nmse_lmmse_val, nmse_db_lmmse_val, ssim_lmmse_val)
                write_set_block("TRAINING", mmse_in_train, nmse_in_train, nmse_in_train_db, ssim_in_train,
                                mmse_train, nmse_train, nmse_train_db, ssim_train,
                                mmse_lmmse_train, nmse_lmmse_train, nmse_db_lmmse_train, ssim_lmmse_train)
            else:
                write_set_block("TEST", mmse_in_test, nmse_in_test, nmse_in_test_db, ssim_in_test,
                                mmse_test, nmse_test, nmse_test_db, ssim_test)
                write_set_block("VALIDATION", mmse_in_val, nmse_in_val, nmse_in_val_db, ssim_in_val,
                                mmse_val, nmse_val, nmse_val_db, ssim_val)
                write_set_block("TRAINING", mmse_in_train, nmse_in_train, nmse_in_train_db, ssim_in_train,
                                mmse_train, nmse_train, nmse_train_db, ssim_train)
        print(f'[Save] Final epoch text report -> {report_path}')
    except Exception as e:
        print(f'[Save Warning] Failed to write final_epoch.txt report: {e}')

    print(f'\n[Done] Finished in {time.perf_counter() - t_start:.1f} s')



if __name__ == '__main__':
    main()
