"""
Single-Dataset CNN Channel Estimator (OpenNTN)
===============================================
Train, evaluate, and test a CNNGenerator on ONE SNR split of the OpenNTN dataset.
All data (train / val / test) come from the *same* SNR folder.

Usage
-----
    python train_single_dataset.py --snr 10
    python train_single_dataset.py --snr -5  --input-type prac --epochs 300
    python train_single_dataset.py --snr 0   --save-model --save-dir ./trained_models

Arguments
---------
    --snr         SNR value in dB  (default: 10).  Supported: -10, -5, 0, 5, 10, 15
    --input-type  Noisy estimator used as CNN input (default: prac). Choices: prac, li
    --epochs      Number of training epochs (default: 200)
    --batch-size  Mini-batch size (default: 16)
    --lr          Adam learning rate (default: 1e-4)
    --train-frac  Fraction of data for training (default: 0.70)
    --val-frac    Fraction of data for validation (default: 0.15); remainder = test
    --n-blocks    Number of residual blocks in CNNGenerator (default: 4)
    --save-model  If set, save the trained model weights to disk
    --save-dir    Directory to save model (default: ./trained_models/SNR_{snr}dB_{type})
    --data-root   Root folder of OpenNTN channel data (auto-detected by default)
    --no-gpu      Disable GPU even if available
"""

# ── Standard library ────────────────────────────────────────────────────────
import os
import sys
import time
import argparse

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
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
# Quick-test mode: tiny subset + few epochs (activated by --test-code)
TEST_CODE_N_TRAIN   = 48           # samples used for training in test-code mode
TEST_CODE_N_VAL     = 16           # samples used for validation
TEST_CODE_N_TEST    = 16           # samples used for test
TEST_CODE_EPOCHS    = 5            # epochs to run in test-code mode
DEFAULT_SAVE_MODEL  = False
DEFAULT_SAVE_DIR    = ''           # auto = ./trained_models/SNR_{snr}dB_{input_type}
DEFAULT_DATA_ROOT   = ''           # auto-detected relative to this script
DEFAULT_N_BLOCKS    = 4
# ============================================================================


# ────────────────────────────────────────────────────────────────────────────
# Path setup  (adds project root + JMMD/Domain_Adversarial helpers to sys.path)
# ────────────────────────────────────────────────────────────────────────────
def _setup_paths():
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        this_dir = os.getcwd()

    project_root = os.path.abspath(os.path.join(this_dir, '..'))
    # Insertion order matters: sys.path.insert(0, p) puts each path at the FRONT.
    # The LAST path inserted ends up at position 0 (= highest priority).
    # We want JMMD/helper > Domain_Adversarial/helper > project_root,
    # so insert project_root first, Domain_Adversarial/helper second, JMMD/helper last.
    for p in [project_root,
              os.path.join(project_root, 'Domain_Adversarial', 'helper'),
              os.path.join(project_root, 'JMMD', 'helper')]:   # ← highest priority
        if p not in sys.path:
            sys.path.insert(0, p)
    return this_dir, project_root


THIS_DIR, PROJECT_ROOT = _setup_paths()

# Lazy import of JMMD utilities (requires project_root on sys.path)
from utils_GAN import CNNGenerator                         # JMMD/helper/utils_GAN.py
from utils import minmaxScaler, deMinMax, complx2real      # Domain_Adversarial/helper/utils.py


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────
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
    If the specified file is missing or changed, dynamically searches 
    generatedChan/OpenNTN for any available .mat file.
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
    Load a .mat file and return numpy arrays with flexible key fallback.

    MATLAB layout : (14, 132, N_samples) -> Transposed to: (N_samples, 132, 14)
    """
    mat = scipy.io.loadmat(mat_path)
    
    # 1. H_perfect key resolution
    if 'H_perfect' in mat and mat['H_perfect'].size > 0:
        H_perfect = mat['H_perfect'].T
    elif 'H_perfect_ori' in mat and mat['H_perfect_ori'].size > 0:
        H_perfect = mat['H_perfect_ori'].T
    else:
        # Fallback: pick first 3D complex array
        for k in mat.keys():
            if not k.startswith('__') and isinstance(mat[k], np.ndarray) and mat[k].ndim == 3:
                H_perfect = mat[k].T
                break

    N_samples, n_subc, n_symb = H_perfect.shape

    # 2. H_input key resolution
    if input_type in ['ls', 'ls_ori']:
        # Sparse LS pilot reconstruction into 2D grid
        ls_key = 'H_ls_pilots_ori' if input_type == 'ls_ori' and 'H_ls_pilots_ori' in mat else 'H_ls_pilots'
        if ls_key not in mat:
            for alt in ['H_ls_pilots', 'H_ls_pilots_ori', 'H_LS_comp', 'H_LS_full']:
                if alt in mat and mat[alt].size > 0:
                    ls_key = alt
                    break

        if ls_key not in mat or 'pilot_rows' not in mat or 'pilot_cols' not in mat:
            raise KeyError(f"Unable to reconstruct sparse LS grid. Required keys '{ls_key}', "
                           f"'pilot_rows', 'pilot_cols' not found in {mat_path}")

        H_pilots   = mat[ls_key]                               # Shape [num_pilots, N_samples] or [N_samples, num_pilots]
        pilot_rows = np.squeeze(mat['pilot_rows']).astype(int) - 1  # 1-indexed -> 0-indexed subcarriers
        pilot_cols = np.squeeze(mat['pilot_cols']).astype(int) - 1  # 1-indexed -> 0-indexed symbols

        # Ensure H_pilots is shape [N_samples, num_pilots]
        if H_pilots.shape[0] != N_samples and H_pilots.shape[1] == N_samples:
            H_pilots = H_pilots.T

        # Construct sparse 2D channel grid (non-zero at pilots, zero elsewhere)
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

    H_perfect = H_perfect.astype(np.complex64)
    H_input   = H_input.astype(np.complex64)
    print(f'  Loaded H_perfect {H_perfect.shape}  |  H_input ({input_key}) {H_input.shape}')
    return H_perfect, H_input




def split_indices(N: int, train_frac: float, val_frac: float, seed: int = 1234):
    """Return reproducible (train, val, test) index arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


# ────────────────────────────────────────────────────────────────────────────
# Helpers: structured dtype required by complx2real()
# ────────────────────────────────────────────────────────────────────────────
_STRUCT_DTYPE = np.dtype([('real', np.float32), ('imag', np.float32)])


def to_structured(H: np.ndarray) -> np.ndarray:
    """Convert a plain complex array [N, S, T] to structured dtype."""
    out = np.empty(H.shape, dtype=_STRUCT_DTYPE)
    out['real'] = H.real.astype(np.float32)
    out['imag'] = H.imag.astype(np.float32)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Pre-processing
# ────────────────────────────────────────────────────────────────────────────
def preprocess_batch(H_perf_batch: np.ndarray, H_in_batch: np.ndarray,
                     lower_range: int):
    """
    Convert complex batches to scaled real-valued TF tensors.

    Returns
    -------
    x_scaled : [B, n_subc, n_symb, 2]  scaled noisy estimate
    y_scaled : [B, n_subc, n_symb, 2]  scaled perfect reference
    x_min    : [B, 2]                  per-sample scaling parameters
    x_max    : [B, 2]
    """
    x = complx2real(to_structured(H_in_batch))   # [B, 2, n_subc, n_symb]
    y = complx2real(to_structured(H_perf_batch))

    x = tf.transpose(x, (0, 2, 3, 1))            # [B, n_subc, n_symb, 2]
    y = tf.transpose(y, (0, 2, 3, 1))

    x_scaled, x_min, x_max = minmaxScaler(x, lower_range=lower_range)
    y_scaled, _,    _      = minmaxScaler(y, min_pre=x_min, max_pre=x_max,
                                          lower_range=lower_range)
    return x_scaled, y_scaled, x_min, x_max


# ────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────
def compute_mmse(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    """Mean-squared error between predicted and true complex channels."""
    return float(np.mean(np.abs(H_pred - H_true) ** 2))


def compute_nmse(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    """Normalised MSE (averaged over samples)."""
    num   = np.mean(np.abs(H_pred - H_true) ** 2, axis=(1, 2))  # [N]
    denom = np.mean(np.abs(H_true) ** 2,           axis=(1, 2))  # [N]
    return float(np.mean(num / (denom + 1e-12)))


def compute_nmse_db(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    """NMSE in dB."""
    return 10.0 * np.log10(compute_nmse(H_pred, H_true) + 1e-12)


def compute_ssim_batch(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    """
    SSIM computed on the *magnitude* of the complex channel images.
    Channels are normalised per-sample to [0, 1] before computing SSIM.

    Parameters
    ----------
    H_pred, H_true : [N, n_subc, n_symb] complex numpy arrays.

    Returns
    -------
    Mean SSIM over the batch.
    """
    mag_pred = np.abs(H_pred).astype(np.float32)
    mag_true = np.abs(H_true).astype(np.float32)

    # per-sample normalise to [0, 1] using H_true statistics
    mn    = mag_true.min(axis=(1, 2), keepdims=True)
    mx    = mag_true.max(axis=(1, 2), keepdims=True)
    scale = np.clip(mx - mn, 1e-8, None)
    mag_pred_n = np.clip((mag_pred - mn) / scale, 0.0, 1.0)
    mag_true_n = (mag_true - mn) / scale

    # TF ssim expects [B, H, W, C]
    pred_t = tf.constant(mag_pred_n[..., np.newaxis])
    true_t = tf.constant(mag_true_n[..., np.newaxis])
    ssim_vals = tf_ssim(true_t, pred_t, max_val=1.0)
    return float(tf.reduce_mean(ssim_vals).numpy())


# ────────────────────────────────────────────────────────────────────────────
# Training step (compiled for speed)
# ────────────────────────────────────────────────────────────────────────────
@tf.function
def _train_step(model, x_scaled, y_scaled, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        residual, _  = model(x_scaled, training=True)
        x_corrected  = x_scaled + residual                     # residual learning
        est_loss     = loss_fn(y_scaled, x_corrected)
        reg_loss     = 0.001 * tf.reduce_mean(tf.square(residual))
        total_loss   = est_loss + reg_loss
        if model.losses:
            total_loss += tf.add_n(model.losses)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, est_loss


# ────────────────────────────────────────────────────────────────────────────
# ONNX Model Export (Architecture + Weights for MATLAB & Python)
# ────────────────────────────────────────────────────────────────────────────
def export_model_to_onnx(model: tf.keras.Model, save_path: str,
                         input_shape=(1, 132, 14, 2)):
    """
    Export full Keras model (architecture + learned weight parameters) to ONNX format.

    The resulting .onnx file contains both the model architecture and trained weights.
    It can be loaded in:
      - Python: via `onnxruntime` or `onnx`
      - MATLAB: via `importONNXNetwork('best.onnx', 'OutputLayerType', 'regression')`

    Parameters
    ----------
    model       : Built tf.keras.Model / CNNGenerator instance
    save_path   : Target filepath for the .onnx model (e.g. 'best.onnx')
    input_shape : Input tensor shape tuple (batch, n_subc, n_symb, 2)
    """
    try:
        import tf2onnx
    except ImportError:
        print('[ONNX] Installing tf2onnx package for ONNX model export ...')
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tf2onnx'])
        import tf2onnx

    try:
        # Build concrete tensor spec for conversion
        spec = (tf.TensorSpec(input_shape, tf.float32, name='input_channel'),)
        model_proto, _ = tf2onnx.convert.from_keras(
            model, input_signature=spec, output_path=save_path)
        print(f'[ONNX Export] Saved ONNX model (architecture + weights) -> {save_path}')
    except Exception as e:
        print(f'[ONNX Export Warning] Failed to export ONNX model: {e}')


# ────────────────────────────────────────────────────────────────────────────
# Inference: produce corrected complex channel
# ────────────────────────────────────────────────────────────────────────────
def infer_channel(model: CNNGenerator,
                  H_perf: np.ndarray, H_in: np.ndarray,
                  batch_size: int, lower_range: int) -> np.ndarray:
    """
    Run the trained model on a full dataset and return the predicted
    complex channel of shape [N, n_subc, n_symb].

    Parameters
    ----------
    model        : trained CNNGenerator
    H_perf       : [N, n_subc, n_symb] complex - perfect channel (used only for
                   computing per-sample minmax scaling reference)
    H_in         : [N, n_subc, n_symb] complex - noisy input channel
    batch_size   : inference batch size
    lower_range  : scaling flag used during training (-1 or 0)

    Returns
    -------
    H_pred : [N, n_subc, n_symb] complex numpy array
    """
    N     = H_perf.shape[0]
    steps = N // batch_size
    out   = []

    for i in range(steps):
        sl   = slice(i * batch_size, (i + 1) * batch_size)
        x_sc, _, x_min, x_max = preprocess_batch(H_perf[sl], H_in[sl], lower_range)
        residual, _ = model(x_sc, training=False)
        x_corr      = x_sc + residual
        x_denorm    = deMinMax(x_corr, x_min, x_max, lower_range=lower_range)
        x_np        = x_denorm.numpy()          # [B, n_subc, n_symb, 2]
        out.append(x_np[..., 0] + 1j * x_np[..., 1])

    # Remainder batch
    if N % batch_size:
        sl   = slice(steps * batch_size, N)
        x_sc, _, x_min, x_max = preprocess_batch(H_perf[sl], H_in[sl], lower_range)
        residual, _ = model(x_sc, training=False)
        x_corr      = x_sc + residual
        x_denorm    = deMinMax(x_corr, x_min, x_max, lower_range=lower_range)
        x_np        = x_denorm.numpy()
        out.append(x_np[..., 0] + 1j * x_np[..., 1])

    return np.concatenate(out, axis=0)   # [N, n_subc, n_symb]


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    # ── Argument parsing ────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='Single-dataset CNN channel estimator for OpenNTN data.',
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
                        help='Directory to save model. '
                             'Default: ./trained_models/SNR_{snr}dB_{input_type}')
    parser.add_argument('--data-root', type=str, default=DEFAULT_DATA_ROOT,
                        help='Root folder for OpenNTN channel data '
                             '(default: auto-detect)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU even if available')
    parser.add_argument('--test-code', action='store_true',
                        help='Quick smoke-test: use a tiny data subset '
                             f'({TEST_CODE_N_TRAIN}/{TEST_CODE_N_VAL}/{TEST_CODE_N_TEST} '
                             f'train/val/test samples) and only '
                             f'{TEST_CODE_EPOCHS} epochs')

    args = parser.parse_args()

    # ── Validate ────────────────────────────────────────────────────────────
    if args.train_frac + args.val_frac >= 1.0:
        parser.error('--train-frac + --val-frac must be < 1.0 (remainder is test).')

    # ── GPU setup ────────────────────────────────────────────────────────────
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

    # ── Config banner ────────────────────────────────────────────────────────
    test_frac = 1.0 - args.train_frac - args.val_frac
    print('\n' + '=' * 58)
    print('  Single-Dataset CNN Channel Estimator  (OpenNTN)')
    print('=' * 58)
    print(f'  SNR           : {args.snr:+d} dB')
    print(f'  Input type    : {args.input_type}')
    print(f'  Epochs        : {args.epochs}')
    print(f'  Batch size    : {args.batch_size}')
    print(f'  Learning rate : {args.lr}')
    print(f'  Split         : {args.train_frac:.0%} / {args.val_frac:.0%} '
          f'/ {test_frac:.0%}  (train / val / test)')
    print(f'  CNN n_blocks  : {args.n_blocks}')
    print('=' * 58 + '\n')

    # ── Data loading & splitting ─────────────────────────────────────────────
    mat_path = get_data_path(args.data_root, args.snr)
    print(f'[Data] {mat_path}')
    H_perfect, H_input = load_mat_data(mat_path, args.input_type)
    N = H_perfect.shape[0]

    idx_train, idx_val, idx_test = split_indices(
        N, args.train_frac, args.val_frac, seed=1234)

    # ── Test-code mode: shrink to a tiny subset ─────────────────────────────
    if args.test_code:
        print('[test-code] Limiting dataset to a tiny subset for quick testing.')
        idx_train = idx_train[:TEST_CODE_N_TRAIN]
        idx_val   = idx_val  [:TEST_CODE_N_VAL]
        idx_test  = idx_test [:TEST_CODE_N_TEST]
        args.epochs = TEST_CODE_EPOCHS
        print(f'[test-code] Epochs overridden to {args.epochs}')

    print(f'[Data] N={N}  '
          f'train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}')

    # ── Model, optimiser, loss ───────────────────────────────────────────────
    model     = CNNGenerator(n_blocks=args.n_blocks)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr,
                                         beta_1=0.5, beta_2=0.9)
    loss_fn   = tf.keras.losses.MeanSquaredError()

    # ── Save directory ───────────────────────────────────────────────────────
    if args.save_dir:
        save_dir = os.path.abspath(args.save_dir)
    else:
        save_dir = os.path.join(THIS_DIR, 'trained_models',
                                f'SNR_{args.snr}dB_{args.input_type}')
    os.makedirs(save_dir, exist_ok=True)
    print(f'[Save] Model dir : {save_dir}\n')

    # ── Training ──────────────────────────────────────────────────────────────
    lower_range     = DEFAULT_LOWER_RANGE
    n_train_batches = len(idx_train) // args.batch_size
    n_val_batches   = len(idx_val)   // args.batch_size

    best_val_loss = float('inf')
    best_epoch    = 0
    history = {
        'train_loss': [],
        'val_loss':   [],
        'val_nmse':   [],
    }

    print(f'[Train] {args.epochs} epochs  |  {n_train_batches} batches/epoch\n')
    t_start = time.perf_counter()

    for epoch in range(args.epochs):
        # Shuffle training data each epoch with a unique seed
        idx_e = np.random.default_rng(epoch).permutation(idx_train)

        # ---------- Training pass ----------
        ep_train_loss = 0.0
        for b in range(n_train_batches):
            batch_idx = idx_e[b * args.batch_size:(b + 1) * args.batch_size]
            h_p = H_perfect[batch_idx]
            h_i = H_input[batch_idx]
            x_sc, y_sc, _, _ = preprocess_batch(h_p, h_i, lower_range)
            total_l, _ = _train_step(model, x_sc, y_sc, optimizer, loss_fn)
            ep_train_loss += total_l.numpy()

        avg_train_loss = ep_train_loss / max(n_train_batches, 1)

        # ---------- Validation pass ----------
        ep_val_loss = 0.0
        ep_val_nmse = 0.0
        for b in range(n_val_batches):
            batch_idx = idx_val[b * args.batch_size:(b + 1) * args.batch_size]
            h_p = H_perfect[batch_idx]
            h_i = H_input[batch_idx]
            x_sc, y_sc, x_min, x_max = preprocess_batch(h_p, h_i, lower_range)
            residual, _ = model(x_sc, training=False)
            x_corr      = x_sc + residual
            ep_val_loss += loss_fn(y_sc, x_corr).numpy()

            # Quick NMSE in scaled domain
            x_np = x_corr.numpy()
            y_np = y_sc.numpy()
            diff_sq = np.mean((x_np - y_np) ** 2, axis=(1, 2, 3))
            ref_sq  = np.mean(y_np ** 2,           axis=(1, 2, 3))
            ep_val_nmse += float(np.mean(diff_sq / (ref_sq + 1e-12)))

        avg_val_loss = ep_val_loss / max(n_val_batches, 1)
        avg_val_nmse = ep_val_nmse / max(n_val_batches, 1)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_nmse'].append(avg_val_nmse)

        # ── Print every 10 epochs (and first / last) ──
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            elapsed = time.perf_counter() - t_start
            print(f'Epoch [{epoch+1:>4d}/{args.epochs}]  '
                  f'TrainLoss={avg_train_loss:.6f}  '
                  f'ValLoss={avg_val_loss:.6f}  '
                  f'ValNMSE={avg_val_nmse:.6f}  '
                  f't={elapsed:.1f}s')

        # ── Save best model ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch    = epoch + 1
            if args.save_model:
                export_model_to_onnx(model, os.path.join(save_dir, 'best.onnx'))
                # Write detailed training configuration & metadata file alongside the weights
                with open(os.path.join(save_dir, 'best_epoch.txt'), 'w') as fh:
                    fh.write(f'best_epoch       = {best_epoch}\n'
                             f'best_val_loss    = {best_val_loss:.8f}\n'
                             f'snr              = {args.snr} dB\n'
                             f'input_type       = {args.input_type}\n'
                             f'total_epochs     = {args.epochs}\n'
                             f'batch_size       = {args.batch_size}\n'
                             f'learning_rate    = {args.lr}\n'
                             f'n_blocks         = {args.n_blocks}\n'
                             f'total_samples    = {N}\n'
                             f'n_train_samples  = {len(idx_train)}\n'
                             f'n_val_samples    = {len(idx_val)}\n'
                             f'n_test_samples   = {len(idx_test)}\n'
                             f'test_code_mode   = {args.test_code}\n'
                             f'data_path        = {mat_path}\n')

    # ── Save final model ──────────────────────────────────────────────────────
    if args.save_model:
        export_model_to_onnx(model, os.path.join(save_dir, 'final.onnx'))
        print(f'\n[Save] Final ONNX model -> {os.path.join(save_dir, "final.onnx")}')
        print(f'[Save] Best  ONNX model -> {os.path.join(save_dir, "best.onnx")}'
              f'  (epoch {best_epoch})')


    # Save training history (.mat)
    hist_path = os.path.join(save_dir, 'training_history.mat')
    scipy.io.savemat(hist_path, {
        'train_loss': np.array(history['train_loss']),
        'val_loss':   np.array(history['val_loss']),
        'val_nmse':   np.array(history['val_nmse']),
        'snr':        args.snr,
        'input_type': args.input_type,
        'n_epochs':   args.epochs,
        'best_epoch': best_epoch,
    })
    print(f'[Save] Training history -> {hist_path}')

    # ── Evaluation on VALIDATION set ─────────────────────────────────────────
    print('\n' + '─' * 58)
    print('[Eval] Validation set ...')
    H_pred_val = infer_channel(model,
                               H_perfect[idx_val], H_input[idx_val],
                               args.batch_size, lower_range)

    mmse_val = compute_mmse(H_pred_val, H_perfect[idx_val])
    nmse_val = compute_nmse(H_pred_val, H_perfect[idx_val])
    ssim_val = compute_ssim_batch(H_pred_val, H_perfect[idx_val])
    nmse_val_db = 10.0 * np.log10(nmse_val + 1e-12)

    # Raw noisy baseline on validation
    mmse_in_val = compute_mmse(H_input[idx_val], H_perfect[idx_val])
    nmse_in_val = compute_nmse(H_input[idx_val], H_perfect[idx_val])
    ssim_in_val = compute_ssim_batch(H_input[idx_val], H_perfect[idx_val])
    nmse_in_val_db = 10.0 * np.log10(nmse_in_val + 1e-12)

    # ── Evaluation on TEST set ────────────────────────────────────────────────
    print('[Eval] Test set ...')
    H_pred_test = infer_channel(model,
                                H_perfect[idx_test], H_input[idx_test],
                                args.batch_size, lower_range)

    mmse_test = compute_mmse(H_pred_test, H_perfect[idx_test])
    nmse_test = compute_nmse(H_pred_test, H_perfect[idx_test])
    ssim_test = compute_ssim_batch(H_pred_test, H_perfect[idx_test])
    nmse_test_db = 10.0 * np.log10(nmse_test + 1e-12)

    # Raw noisy baseline on test
    mmse_in_test = compute_mmse(H_input[idx_test], H_perfect[idx_test])
    nmse_in_test = compute_nmse(H_input[idx_test], H_perfect[idx_test])
    ssim_in_test = compute_ssim_batch(H_input[idx_test], H_perfect[idx_test])
    nmse_in_test_db = 10.0 * np.log10(nmse_in_test + 1e-12)

    # ── Results table ─────────────────────────────────────────────────────────
    hdr = f'  SNR={args.snr:+d} dB  |  input type: {args.input_type}'
    print('\n' + '═' * 60)
    print('  EVALUATION RESULTS')
    print('═' * 60)
    print(hdr)
    print('─' * 60)
    print(f'  {"Metric":<14} {"Raw "+args.input_type:<20} {"CNN output":<20}')
    print('─' * 60)
    print(f'  {"[VAL]":<14}')
    print(f'  {"  MMSE":<14} {mmse_in_val:<20.6e} {mmse_val:<20.6e}')
    print(f'  {"  NMSE":<14} {nmse_in_val:<20.6f} {nmse_val:<20.6f}')
    print(f'  {"  NMSE(dB)":<14} {nmse_in_val_db:<20.2f} {nmse_val_db:<20.2f}')
    print(f'  {"  SSIM":<14} {ssim_in_val:<20.6f} {ssim_val:<20.6f}')
    print('─' * 60)
    print(f'  {"[TEST]":<14}')
    print(f'  {"  MMSE":<14} {mmse_in_test:<20.6e} {mmse_test:<20.6e}')
    print(f'  {"  NMSE":<14} {nmse_in_test:<20.6f} {nmse_test:<20.6f}')
    print(f'  {"  NMSE(dB)":<14} {nmse_in_test_db:<20.2f} {nmse_test_db:<20.2f}')
    print(f'  {"  SSIM":<14} {ssim_in_test:<20.6f} {ssim_test:<20.6f}')
    print('═' * 60)

    # ── Save evaluation .mat ──────────────────────────────────────────────────
    eval_path = os.path.join(save_dir, 'evaluation_results.mat')
    scipy.io.savemat(eval_path, {
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
    })
    print(f'[Save] Evaluation results -> {eval_path}')
    print(f'\n[Done] Finished in {time.perf_counter() - t_start:.1f} s')


# ────────────────────────────────────────────────────────────────────────────
# HOW TO RELOAD A SAVED MODEL
# ────────────────────────────────────────────────────────────────────────────
def load_trained_model(save_dir: str,
                       n_blocks: int = DEFAULT_N_BLOCKS,
                       use_best: bool = True) -> CNNGenerator:
    """
    Reload a previously trained CNNGenerator from disk.

    Parameters
    ----------
    save_dir : path that was printed / saved during training
               e.g. './single_dataset/trained_models/SNR_10dB_prac'
    n_blocks : must match the value used during training (default: 4)
    use_best : load 'best' weights if True, 'final' weights if False

    Example
    -------
    >>> from train_single_dataset import load_trained_model, infer_channel
    >>> model = load_trained_model('./trained_models/SNR_10dB_prac')
    >>> H_pred = infer_channel(model, H_perfect_test, H_input_test,
    ...                        batch_size=16, lower_range=-1)
    """
    filename = 'best.weights.h5' if use_best else 'final.weights.h5'
    ckpt     = os.path.join(save_dir, filename)
    if not os.path.exists(ckpt):
        fallback = os.path.join(save_dir, 'best' if use_best else 'final')
        if os.path.exists(fallback) or os.path.exists(fallback + '.index'):
            ckpt = fallback

    # Build the model graph before loading weights
    dummy = tf.zeros((1, 132, 14, 2))
    model(dummy, training=False)

    model.load_weights(ckpt)
    print(f'[Reload] {"best" if use_best else "final"} weights loaded from: {ckpt}')
    return model


if __name__ == '__main__':
    main()
