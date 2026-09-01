"""
====================================================================================================
CORAL cGAN Domain Adaptation for NTN Channel Estimation (OpenNTN)
====================================================================================================

Overview
--------
This script trains and evaluates a Conditional Generative Adversarial Network (cGAN) based on 
Pix2Pix (UNet Generator + PatchGAN Discriminator with WGAN-GP) to perform Unsupervised Domain 
Adaptation (UDA) for 5G Non-Terrestrial Network (NTN) channel estimation.

Domain shift (e.g., between different user speeds, propagation delays, or TDL channel profiles)
is mitigated using Correlation Alignment (CORAL) loss, which aligns the second-order statistics
(covariance matrices) of intermediate feature representations between the source and target domains.

Dataset Splitting (3-Way Split)
--------------------------------
- Train Set (Default 70%): Labeled source domain + Unlabeled target domain for CORAL adaptation.
- Validation Set (Default 15%): Periodic evaluation and model checkpointing during training.
- Test Set (Default 15%): Final held-out evaluation exported to `testChannel_source.mat` and 
  `testChannel_target.mat` for subsequent BER simulations and benchmark comparisons.

Usage
-----
    # Default direct estimation cGAN with CORAL adaptation (70/15/15 split)
    python run_CORAL_cGAN.py --type LI --domain-weight 0.5

    # Source-only baseline (no domain adaptation)
    python run_CORAL_cGAN.py --type LI --only-source

    # Residual learning mode with hybrid CORAL loss
    python run_CORAL_cGAN.py --type LI --residual --coral-type hybrid

    # Quick test run (small subset, 5 epochs)
    python run_CORAL_cGAN.py --type LI --test-code
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

import tensorflow as tf
import os
import sys
import re
import scipy
from scipy.io import savemat
import h5py
import time
import argparse

# ============================================================================
# CONFIGURATION CONSTANTS
# Edit these paths and parameters directly to run from your IDE
# ============================================================================
SOURCE_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps"
TARGET_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps"
SAVE_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\CORAL\model"
MODEL_TYPE = "LI"              # "LI", "LS", or "Prac"
ONLY_SOURCE = False             # Set True to train only on source (no CORAL)
RESIDUAL = False                # Set True for residual cGAN, False for direct estimation cGAN
CORAL_TYPE = "global_pooling"   # "global_pooling" or "hybrid"
DEFAULT_TRAIN_FRAC = 0.70       # Fraction of data for training
DEFAULT_VAL_FRAC = 0.15         # Fraction of data for validation (remaining is test)
N_EPOCHS = 300                  # Number of epochs
BATCH_SIZE = 16                 # Batch size
TEST_CODE = False               # Run with subset of data (size 96, 5 epochs) for testing
# ============================================================================

# Setup project directories and paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
helper_dir = os.path.join(project_root, 'JMMD', 'helper')
domain_helper_dir = os.path.join(project_root, 'Domain_Adversarial', 'helper')

for path in [project_root, helper_dir, domain_helper_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import helper functions
from Domain_Adversarial.helper import loader, plotfig, PAD
from Domain_Adversarial.helper.utils import H5BatchLoader, minmaxScaler, complx2real, deMinMax

# WGAN-GP + CORAL helpers
from JMMD.helper.utils_GAN import (
    GAN,
    GlobalPoolingCORALLoss,
    HybridCORALLoss,
    train_step_wgan_gp_coral,
    val_step_wgan_gp_coral,
    train_step_wgan_gp_coral_residual,
    val_step_wgan_gp_coral_residual,
    train_step_wgan_gp_source_only,
    val_step_wgan_gp_source_only,
    train_step_wgan_gp_source_only_residual,
    val_step_wgan_gp_source_only_residual,
    post_val,
    visualize_H,
    save_checkpoint_jmmd as save_checkpoint,
    WeightScheduler
)


def split_indices(N: int, train_frac: float, val_frac: float, seed: int = 1234):
    """Return reproducible (train, val, test) index arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def infer_and_evaluate(generator, loader_true, loader_input, lower_range=-1, is_residual=False):
    """
    Run full-dataset inference over H5BatchLoader instances and calculate metrics.
    """
    loader_true.reset()
    loader_input.reset()

    all_preds = []
    total_samples = 0
    total_se = 0.0
    total_pe = 0.0

    for _ in range(loader_input.num_batches):
        H_true_batch = next(loader_true.generator())
        H_in_batch = next(loader_input.generator())

        # Scale input
        H_in_scaled, params = minmaxScaler(H_in_batch, lower_range=lower_range)
        H_in_real = complx2real(H_in_scaled)

        # Generator prediction
        pred_real = generator(H_in_real, training=False)
        pred_scaled = pred_real[..., 0] + 1j * pred_real[..., 1]

        # Descale
        pred_descaled = deMinMax(pred_scaled, params, lower_range=lower_range)

        # Add residual if enabled
        if is_residual:
            H_est_batch = H_in_batch + pred_descaled
        else:
            H_est_batch = pred_descaled

        all_preds.append(H_est_batch)

        # Accumulate squared errors
        err = H_true_batch - H_est_batch
        se = np.sum(np.abs(err) ** 2)
        pe = np.sum(np.abs(H_true_batch) ** 2)

        total_se += se
        total_pe += pe
        total_samples += H_true_batch.shape[0]

    H_pred_all = np.concatenate(all_preds, axis=0) if all_preds else np.empty((0,))

    nmse = float(total_se / (total_pe + 1e-12)) if total_pe > 0 else 0.0
    nmse_db = float(10.0 * np.log10(nmse + 1e-12)) if nmse > 0 else 0.0
    mmse = float(total_se / (total_samples * H_true_batch.shape[1] * H_true_batch.shape[2])) if total_samples > 0 else 0.0

    metrics = {
        'nmse': nmse,
        'nmse_db': nmse_db,
        'mmse': mmse
    }
    return H_pred_all, metrics


def save_test_channel_mat(mat_file_obj, test_indices, H_pred_test, save_filepath, snr_str, model_type_tag):
    """
    Save test channel grids matching the train_cGAN.py schema into testChannel_*.mat.
    """
    test_dict = {}

    # 1. H_perfect_test (Effective channel)
    if 'H_perfect' in mat_file_obj:
        test_dict['H_perfect_test'] = mat_file_obj['H_perfect'][test_indices]

    # 2. H_original_test (Channel before Doppler compensation if available)
    if 'H_perfect_ori' in mat_file_obj:
        test_dict['H_original_test'] = mat_file_obj['H_perfect_ori'][test_indices]
    elif 'H_perfect' in mat_file_obj:
        test_dict['H_original_test'] = mat_file_obj['H_perfect'][test_indices]

    # 3. H_LS_test (LS pilot sequence or initial estimation grid)
    if 'H_ls_pilots' in mat_file_obj:
        test_dict['H_LS_test'] = mat_file_obj['H_ls_pilots'][test_indices]
    elif f'H_{model_type_tag.lower()}' in mat_file_obj:
        test_dict['H_LS_test'] = mat_file_obj[f'H_{model_type_tag.lower()}'][test_indices]

    # 4. Pilot coordinates (1-indexed for MATLAB compatibility)
    if 'pilot_rows' in mat_file_obj:
        p_rows = np.squeeze(mat_file_obj['pilot_rows'][()])
        test_dict['pilot_rows'] = p_rows + 1 if np.min(p_rows) == 0 else p_rows
    if 'pilot_cols' in mat_file_obj:
        p_cols = np.squeeze(mat_file_obj['pilot_cols'][()])
        test_dict['pilot_cols'] = p_cols + 1 if np.min(p_cols) == 0 else p_cols

    # 5. Benchmark LI Channel Grid if present
    if 'H_li' in mat_file_obj:
        test_dict['H_LI_test'] = mat_file_obj['H_li'][test_indices]

    # 6. Model Output on Test Set
    test_dict['H_output_test'] = H_pred_test
    test_dict['test_indices'] = test_indices
    test_dict['snr'] = snr_str
    test_dict['model_type'] = model_type_tag

    savemat(save_filepath, test_dict)
    print(f"[Save] Saved test channel grids MAT file -> {save_filepath}")


def main():
    parser = argparse.ArgumentParser(description="CORAL WGAN-GP Domain Adaptation runner for cGAN.")
    parser.add_argument('--source-dir', type=str, default=SOURCE_DIR, help="Source dataset directory containing matlabNTN.mat")
    parser.add_argument('--target-dir', type=str, default=TARGET_DIR, help="Target dataset directory containing matlabNTN.mat")
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR, help="Base folder directory to save results")
    parser.add_argument('--type', type=str, default=MODEL_TYPE, choices=['LI', 'LS', 'Prac'], help="Type tag mapping to sub_folder")
    parser.add_argument('--only-source', action='store_true', default=ONLY_SOURCE, help="Train using source-only data (no CORAL)")
    parser.add_argument('--residual', action='store_true', default=RESIDUAL, help="Use residual learning for cGAN")
    parser.add_argument('--coral-type', type=str, default=CORAL_TYPE, choices=['global_pooling', 'hybrid'], help="CORAL loss variant")
    parser.add_argument('--train-frac', type=float, default=DEFAULT_TRAIN_FRAC, help="Fraction of data for training (default: 0.70)")
    parser.add_argument('--val-frac', type=float, default=DEFAULT_VAL_FRAC, help="Fraction of data for validation (default: 0.15)")
    parser.add_argument('--n-epochs', type=int, default=N_EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('--test-code', action='store_true', default=TEST_CODE, help="Run with subset of data for testing")
    parser.add_argument('--norm-approach', type=str, default='minmax', choices=['minmax', 'std', 'no'], help="Normalization approach")
    parser.add_argument('--lower-range', type=int, default=-1, choices=[0, -1], help="Scaling range for minmax")
    parser.add_argument('--adv-weight', type=float, default=0.005, help="GAN adversarial loss weight")
    parser.add_argument('--est-weight', type=float, default=1.0, help="Estimation loss weight")
    parser.add_argument('--domain-weight', type=float, default=0.5, help="CORAL loss weight")
    parser.add_argument('--temporal-weight', type=float, default=0.02, help="Temporal smoothness weight")
    parser.add_argument('--frequency-weight', type=float, default=0.1, help="Frequency smoothness weight")
    parser.add_argument('--save-model', action='store_true', help="Save trained model checkpoints")

    args = parser.parse_args()

    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError(f"train_frac ({args.train_frac}) + val_frac ({args.val_frac}) must be < 1.0 to leave room for test set.")
    test_frac = 1.0 - args.train_frac - args.val_frac

    sub_folder_map = {'LI': 'GAN_linear', 'LS': 'GAN_ls', 'Prac': 'GAN_practical'}
    sub_folder = sub_folder_map[args.type]

    # If source only, domain weight is always 0.0
    domain_weight = 0.0 if args.only_source else args.domain_weight

    weights = {
        'adv_weight': args.adv_weight,
        'est_weight': args.est_weight,
        'domain_weight': domain_weight,
        'temporal_weight': args.temporal_weight,
        'frequency_weight': args.frequency_weight
    }

    # Initialize CORAL loss function
    if args.coral_type == 'hybrid':
        coral_loss_fn = HybridCORALLoss()
    else:
        coral_loss_fn = GlobalPoolingCORALLoss()

    # Resolve dataset paths dynamically
    def get_mat_file(dir_path):
        default_path = os.path.join(dir_path, 'matlabNTN.mat')
        if os.path.exists(default_path):
            return default_path
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            mat_files = [f for f in os.listdir(dir_path) if f.endswith('.mat') and not f.startswith('inferredChannel')]
            if mat_files:
                return os.path.join(dir_path, mat_files[0])
        return default_path

    source_data_file_path = os.path.abspath(get_mat_file(args.source_dir))
    target_data_file_path = os.path.abspath(get_mat_file(args.target_dir))

    mode_str = 'Source-Only' if args.only_source else f'CORAL ({args.coral_type}) Domain Adaptation'
    arch_str = 'Residual cGAN' if args.residual else 'Direct cGAN'
    print(f"Mode: {mode_str} | Architecture: {arch_str}")
    print(f"Loading Source dataset: {source_data_file_path}")
    print(f"Loading Target dataset: {target_data_file_path}")
    print(f"Split: {args.train_frac:.0%} Train / {args.val_frac:.0%} Val / {test_frac:.0%} Test")

    # Build save path
    base_save_dir = os.path.abspath(args.save_dir)

    def get_snr_db(dir_path):
        match = re.search(r'SNR_[-+]?\d+dB', dir_path, re.IGNORECASE)
        if match:
            return match.group()
        match_num = re.search(r'[-+]?\d+', os.path.basename(dir_path))
        if match_num:
            return f"{match_num.group()}_dB"
        return "unknown_dB"

    def get_tdl_name(dir_path):
        normalized = os.path.normpath(dir_path)
        parts = normalized.split(os.sep)
        for part in reversed(parts):
            if 'tdl' in part.lower():
                return part.replace('TDL_', '').replace('_sim', '').replace('_simple', '').replace('_', '')
        return os.path.basename(os.path.dirname(dir_path))

    src_name = get_tdl_name(args.source_dir)
    tgt_name = get_tdl_name(args.target_dir)
    tgt_snr = get_snr_db(args.target_dir)

    prefix = 'GAN_onlySource' if args.only_source else 'GAN_coral'
    if args.residual:
        prefix += '_residual'
    path_temp = os.path.join(base_save_dir, f'{prefix}_{src_name}_{tgt_name}', tgt_snr)

    os.makedirs(path_temp, exist_ok=True)
    idx_save_path = loader.find_incremental_filename(path_temp, 'ver', '_', '')
    model_path = os.path.join(path_temp, f'ver{idx_save_path}_')

    print(f"Model outputs will be saved to: {model_path}")

    # ============ Load Source and Target data ==============
    source_file = h5py.File(source_data_file_path, 'r')
    H_true_source = source_file['H_perfect']
    N_samp_source = H_true_source.shape[0]

    target_file = h5py.File(target_data_file_path, 'r')
    H_true_target = target_file['H_perfect']
    N_samp_target = H_true_target.shape[0]

    # Perform reproducible 3-Way Split (Train / Val / Test)
    if args.test_code:
        indices_train_source = np.arange(0, 64)
        indices_val_source = np.arange(64, 80)
        indices_test_source = np.arange(80, 96)

        indices_train_target = np.arange(0, 64)
        indices_val_target = np.arange(64, 80)
        indices_test_target = np.arange(80, 96)

        args.n_epochs = 5
    else:
        indices_train_src, indices_val_src, indices_test_src = split_indices(
            N_samp_source, args.train_frac, args.val_frac, seed=1234
        )
        indices_train_tgt, indices_val_tgt, indices_test_tgt = split_indices(
            N_samp_target, args.train_frac, args.val_frac, seed=1234
        )

        # Align training size to batch size
        n_train_src = (len(indices_train_src) // args.batch_size) * args.batch_size
        n_val_src = (len(indices_val_src) // args.batch_size) * args.batch_size
        n_test_src = (len(indices_test_src) // args.batch_size) * args.batch_size

        n_train_tgt = (len(indices_train_tgt) // args.batch_size) * args.batch_size
        n_val_tgt = (len(indices_val_tgt) // args.batch_size) * args.batch_size
        n_test_tgt = (len(indices_test_tgt) // args.batch_size) * args.batch_size

        indices_train_source = indices_train_src[:n_train_src]
        indices_val_source = indices_val_src[:n_val_src]
        indices_test_source = indices_test_src[:n_test_src]

        indices_train_target = indices_train_tgt[:n_train_tgt]
        indices_val_target = indices_val_tgt[:n_val_tgt]
        indices_test_target = indices_test_tgt[:n_test_tgt]

    print(f"Source split -> Train: {len(indices_train_source)} | Val: {len(indices_val_source)} | Test: {len(indices_test_source)}")
    print(f"Target split -> Train: {len(indices_train_target)} | Val: {len(indices_val_target)} | Test: {len(indices_test_target)}")

    class DataLoaders:
        def __init__(self, file, indices_train, indices_val, indices_test, tag, batch_size):
            self.true_train = H5BatchLoader(file, dataset_name='H_perfect', batch_size=batch_size, shuffled_indices=indices_train)
            self.true_val = H5BatchLoader(file, dataset_name='H_perfect', batch_size=batch_size, shuffled_indices=indices_val)
            self.true_test = H5BatchLoader(file, dataset_name='H_perfect', batch_size=batch_size, shuffled_indices=indices_test)
            self.input_train = H5BatchLoader(file, f'H_{tag}', batch_size=batch_size, shuffled_indices=indices_train)
            self.input_val = H5BatchLoader(file, f'H_{tag}', batch_size=batch_size, shuffled_indices=indices_val)
            self.input_test = H5BatchLoader(file, f'H_{tag}', batch_size=batch_size, shuffled_indices=indices_test)

    class_dict_source = {
        'GAN_practical': DataLoaders(source_file, indices_train_source, indices_val_source, indices_test_source, 'prac', args.batch_size),
        'GAN_linear': DataLoaders(source_file, indices_train_source, indices_val_source, indices_test_source, 'li', args.batch_size),
        'GAN_ls': DataLoaders(source_file, indices_train_source, indices_val_source, indices_test_source, 'ls', args.batch_size)
    }

    class_dict_target = {
        'GAN_practical': DataLoaders(target_file, indices_train_target, indices_val_target, indices_test_target, 'prac', args.batch_size),
        'GAN_linear': DataLoaders(target_file, indices_train_target, indices_val_target, indices_test_target, 'li', args.batch_size),
        'GAN_ls': DataLoaders(target_file, indices_train_target, indices_val_target, indices_test_target, 'ls', args.batch_size)
    }

    loss_fn_ce = tf.keras.losses.MeanSquaredError()
    loss_fn_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    start_time = time.perf_counter()

    loader_H_true_train_source = class_dict_source[sub_folder].true_train
    loader_H_input_train_source = class_dict_source[sub_folder].input_train
    loader_H_true_val_source = class_dict_source[sub_folder].true_val
    loader_H_input_val_source = class_dict_source[sub_folder].input_val
    loader_H_true_test_source = class_dict_source[sub_folder].true_test
    loader_H_input_test_source = class_dict_source[sub_folder].input_test

    loader_H_true_train_target = class_dict_target[sub_folder].true_train
    loader_H_input_train_target = class_dict_target[sub_folder].input_train
    loader_H_true_val_target = class_dict_target[sub_folder].true_val
    loader_H_input_val_target = class_dict_target[sub_folder].input_val
    loader_H_true_test_target = class_dict_target[sub_folder].true_test
    loader_H_input_test_target = class_dict_target[sub_folder].input_test

    # Create Distribution Plots before training
    os.makedirs(os.path.join(model_path, sub_folder, "Distribution"), exist_ok=True)
    plotfig.plotHist(loader_H_input_train_source, fig_show=False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='source_beforeTrain', percent=99)
    plotfig.plotHist(loader_H_input_train_target, fig_show=False, save_path=f"{model_path}/{sub_folder}/Distribution/", name='target_beforeTrain', percent=99)

    # Initial metrics computation
    print("Calculating initial metrics...")
    w_dist_epoc = plotfig.wasserstein_approximate(loader_H_input_train_source, loader_H_input_train_target)
    pad_svm = PAD.original_PAD(loader_H_input_train_source, loader_H_input_train_target)
    print(f"Initial SVM PAD = {pad_svm:.4f}")

    X_features_, y_features_ = PAD.extract_features_with_pca(loader_H_input_train_source, loader_H_input_train_target, pca_components=100)
    pad_pca_svm_epoc = PAD.calc_pad_svm(X_features_, y_features_)
    pad_pca_lda_epoc = PAD.calc_pad_lda(X_features_, y_features_)
    pad_pca_logreg_epoc = PAD.calc_pad_logreg(X_features_, y_features_)

    pad_metrics = {
        'pad_pca_lda': {'before_training': pad_pca_lda_epoc},
        'pad_pca_logreg': {'before_training': pad_pca_logreg_epoc},
        'pad_pca_svm': {'before_training': pad_pca_svm_epoc},
        'w_dist': {'before_training': w_dist_epoc}
    }

    # Initialize model
    sample_H = next(loader_H_input_train_source.generator())
    n_subc = sample_H.shape[1]
    print(f"Detected channel subcarriers (n_subc): {n_subc}")

    model = GAN(n_subc=n_subc, gen_l2=None, disc_l2=1e-5)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.9)
    optimizer = [gen_optimizer, disc_optimizer]

    train_metrics = {
        'train_loss': [],
        'train_est_loss': [],
        'train_disc_loss': [],
        'train_domain_loss': [],
        'train_est_loss_target': []
    }

    val_metrics = {
        'val_loss': [],
        'val_gan_disc_loss': [],
        'val_domain_loss': [],
        'val_est_loss_source': [],
        'val_est_loss_target': [],
        'val_est_loss': [],
        'source_acc': [],
        'target_acc': [],
        'acc': [],
        'nmse_val_source': [],
        'nmse_val_target': [],
        'nmse_val': [],
        'val_smoothness_loss': []
    }

    H_to_save = {}
    flag = 1
    epoc_pad = []
    linear_interp = False

    epoch_min = 50 if args.only_source else 20
    epoch_step = 50 if args.only_source else 20

    # Select appropriate training and validation functions
    if args.only_source:
        train_fn = train_step_wgan_gp_source_only_residual if args.residual else train_step_wgan_gp_source_only
        val_fn = val_step_wgan_gp_source_only_residual if args.residual else val_step_wgan_gp_source_only
    else:
        train_fn = train_step_wgan_gp_coral_residual if args.residual else train_step_wgan_gp_coral
        val_fn = val_step_wgan_gp_coral_residual if args.residual else val_step_wgan_gp_coral

    H_val_gen = None

    # Main training loop
    for epoch in range(args.n_epochs):
        loader_H_true_train_source.reset()
        loader_H_input_train_source.reset()
        loader_H_true_train_target.reset()
        loader_H_input_train_target.reset()

        loader_H = [loader_H_input_train_source, loader_H_true_train_source, loader_H_input_train_target, loader_H_true_train_target]
        loss_fn = [loss_fn_ce, loss_fn_bce]

        if epoch in [int(args.n_epochs * r) for r in [0, 0.25, 0.5, 0.75]] or epoch == args.n_epochs - 1:
            return_features = True
            epoc_pad.append(epoch)
        else:
            return_features = False

        # Run training step
        if args.only_source:
            train_step_output = train_fn(
                model, loader_H, loss_fn, optimizer, lower_range=args.lower_range,
                save_features=True, weights=weights, linear_interp=linear_interp
            )
        else:
            train_step_output = train_fn(
                model, loader_H, loss_fn, optimizer, lower_range=args.lower_range,
                coral_loss_fn=coral_loss_fn, save_features=True, weights=weights, linear_interp=linear_interp
            )

        train_epoc_loss_est = train_step_output.avg_epoc_loss_est
        train_epoc_loss_d = train_step_output.avg_epoc_loss_d
        train_epoc_loss_domain = train_step_output.avg_epoc_loss_domain
        train_epoc_loss = train_step_output.avg_epoc_loss
        train_epoc_loss_est_target = train_step_output.avg_epoc_loss_est_target

        print(f"Epoch {epoch+1}/{args.n_epochs} | Time: {time.perf_counter() - start_time:.2f}s")
        print(f"  Loss: {train_epoc_loss:.6f} | Est Loss (Src): {train_epoc_loss_est:.6f} | {'CORAL' if not args.only_source else 'OnlySource'} Loss: {train_epoc_loss_domain:.6f}")

        # Compute PAD metrics for features
        if return_features and (weights['domain_weight'] != 0):
            features_source_file = "features_source.h5"
            features_target_file = "features_target.h5"

            if os.path.exists(features_source_file) and os.path.exists(features_target_file):
                X_features, y_features = PAD.extract_features_with_pca(features_source_file, features_target_file, pca_components=100)
                pad_metrics['pad_pca_svm'][f'epoch_{epoch+1}'] = PAD.calc_pad_svm(X_features, y_features)
                pad_metrics['pad_pca_lda'][f'epoch_{epoch+1}'] = PAD.calc_pad_lda(X_features, y_features)
                pad_metrics['pad_pca_logreg'][f'epoch_{epoch+1}'] = PAD.calc_pad_logreg(X_features, y_features)

                plotfig.plotHist(features_source_file, fig_show=False, save_path=f"{model_path}/{sub_folder}/Distribution/", name=f'source_epoch_{epoch+1}', percent=99)
                plotfig.plotHist(features_target_file, fig_show=False, save_path=f"{model_path}/{sub_folder}/Distribution/", name=f'target_epoch_{epoch+1}', percent=99)

                os.remove(features_source_file)
                os.remove(features_target_file)

        train_metrics['train_loss'].append(train_epoc_loss)
        train_metrics['train_est_loss'].append(train_epoc_loss_est)
        train_metrics['train_disc_loss'].append(train_epoc_loss_d)
        train_metrics['train_domain_loss'].append(train_epoc_loss_domain)
        train_metrics['train_est_loss_target'].append(train_epoc_loss_est_target)

        # ===================== Periodic Validation =====================
        loader_H_true_val_source.reset()
        loader_H_input_val_source.reset()
        loader_H_true_val_target.reset()
        loader_H_input_val_target.reset()
        loader_H_eval = [loader_H_input_val_source, loader_H_true_val_source, loader_H_input_val_target, loader_H_true_val_target]

        is_periodic_eval = (epoch == epoch_min) or (epoch + 1 > epoch_min and (epoch - epoch_min) % epoch_step == 0)

        if is_periodic_eval and epoch != args.n_epochs - 1:
            if args.only_source:
                H_sample, epoc_val_return = val_fn(
                    model, loader_H_eval, loss_fn, args.lower_range, weights=weights, linear_interp=linear_interp
                )
            else:
                H_sample, epoc_val_return = val_fn(
                    model, loader_H_eval, loss_fn, args.lower_range, coral_loss_fn=coral_loss_fn, weights=weights, linear_interp=linear_interp
                )
            visualize_H(H_sample, H_to_save, epoch, plotfig.figChan, flag, model_path, sub_folder, domain_weight=weights['domain_weight'])
            flag = 0
        elif epoch == args.n_epochs - 1:
            if args.only_source:
                _, epoc_val_return, H_val_gen = val_fn(
                    model, loader_H_eval, loss_fn, args.lower_range, weights=weights, linear_interp=linear_interp, return_H_gen=True
                )
            else:
                _, epoc_val_return, H_val_gen = val_fn(
                    model, loader_H_eval, loss_fn, args.lower_range, coral_loss_fn=coral_loss_fn, weights=weights, linear_interp=linear_interp, return_H_gen=True
                )
        else:
            if args.only_source:
                _, epoc_val_return = val_fn(
                    model, loader_H_eval, loss_fn, args.lower_range, weights=weights, linear_interp=linear_interp
                )
            else:
                _, epoc_val_return = val_fn(
                    model, loader_H_eval, loss_fn, args.lower_range, coral_loss_fn=coral_loss_fn, weights=weights, linear_interp=linear_interp
                )

        post_val(epoc_val_return, epoch, args.n_epochs, val_metrics, domain_weight=weights['domain_weight'])

        # Save Checkpoint
        if is_periodic_eval or epoch == args.n_epochs - 1:
            all_metrics = {
                'figLoss': plotfig.figLoss,
                'savemat': savemat,
                'pad_metrics': pad_metrics,
                'epoc_pad': epoc_pad,
                'pad_svm': pad_svm,
                'weights': weights,
                'optimizer': optimizer
            }
            all_metrics.update(train_metrics)
            all_metrics.update(val_metrics)

            save_checkpoint(model, args.save_model, model_path, sub_folder, epoch, all_metrics)

    # Save visual outputs
    os.makedirs(os.path.join(model_path, sub_folder, 'H_visualize'), exist_ok=True)
    if H_to_save:
        savemat(os.path.join(model_path, sub_folder, 'H_visualize', 'H_trix.mat'), H_to_save)
    if H_val_gen is not None:
        savemat(os.path.join(model_path, sub_folder, 'H_visualize', 'H_val_generated.mat'), {
            'H_val_gen': H_val_gen,
            'indices_val_source': indices_val_source,
            'indices_val_target': indices_val_target
        })

    # ============================================================================
    # FINAL TEST EVALUATION & testChannel_*.mat EXPORT
    # ============================================================================
    print("\n" + "=" * 80)
    print("RUNNING FINAL TEST SET EVALUATION")
    print("=" * 80)

    # 1. Evaluate Source Test Set
    H_pred_test_src, metrics_test_src = infer_and_evaluate(
        model.gen, loader_H_true_test_source, loader_H_input_test_source,
        lower_range=args.lower_range, is_residual=args.residual
    )
    print(f"[Source Test] NMSE: {metrics_test_src['nmse_db']:.2f} dB (Linear: {metrics_test_src['nmse']:.6f}) | MMSE: {metrics_test_src['mmse']:.6f}")

    # 2. Evaluate Target Test Set
    H_pred_test_tgt, metrics_test_tgt = infer_and_evaluate(
        model.gen, loader_H_true_test_target, loader_H_input_test_target,
        lower_range=args.lower_range, is_residual=args.residual
    )
    print(f"[Target Test] NMSE: {metrics_test_tgt['nmse_db']:.2f} dB (Linear: {metrics_test_tgt['nmse']:.6f}) | MMSE: {metrics_test_tgt['mmse']:.6f}")

    # 3. Save testChannel_source.mat and testChannel_target.mat
    save_test_channel_mat(
        source_file, indices_test_source, H_pred_test_src,
        os.path.join(model_path, sub_folder, 'testChannel_source.mat'),
        tgt_snr, args.type
    )

    save_test_channel_mat(
        target_file, indices_test_target, H_pred_test_tgt,
        os.path.join(model_path, sub_folder, 'testChannel_target.mat'),
        tgt_snr, args.type
    )

    # 4. Save consolidated evaluation_results.mat
    eval_dict = {
        'nmse_db_test_source': metrics_test_src['nmse_db'],
        'nmse_test_source': metrics_test_src['nmse'],
        'mmse_test_source': metrics_test_src['mmse'],
        'nmse_db_test_target': metrics_test_tgt['nmse_db'],
        'nmse_test_target': metrics_test_tgt['nmse'],
        'mmse_test_target': metrics_test_tgt['mmse'],
        'indices_train_source': indices_train_source,
        'indices_val_source': indices_val_source,
        'indices_test_source': indices_test_source,
        'indices_train_target': indices_train_target,
        'indices_val_target': indices_val_target,
        'indices_test_target': indices_test_target,
        'weights': weights,
        'model_type': args.type,
        'residual': args.residual,
        'coral_type': args.coral_type
    }
    savemat(os.path.join(model_path, sub_folder, 'evaluation_results.mat'), eval_dict)
    print(f"[Save] Saved consolidated metrics -> {os.path.join(model_path, sub_folder, 'evaluation_results.mat')}")

    # Close HDF5 file handles
    source_file.close()
    target_file.close()

    print(f"\nCORAL cGAN Domain Adaptation completed successfully! Results stored in: {model_path}/{sub_folder}")


if __name__ == '__main__':
    main()
