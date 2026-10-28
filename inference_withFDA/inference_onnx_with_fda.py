"""
========================================================================================
Test-Time FDA ONNX Channel Inference Pipeline (Grid & Sequence Models)
========================================================================================
Supports:
  1. Both Model Architectures:
     - Grid Models (e.g. LI_DnCNN, LI_cGAN)
     - Sequence Models (e.g. LS_Attention, LS_Transformer)
  2. Complete Test-Time FDA Pipeline:
     - Stage 'none'     : Standard baseline inference (no FDA)
     - Stage 'pre'      : Pre-FDA on input only (Target input -> Source style)
     - Stage 'post'     : Post-FDA on output only (Model output -> Target style restoration)
     - Stage 'dual'     : Pre-FDA on input + Post-FDA on output (Recommended)
     - Stage 'dual_pcp' : Dual FDA + Pilot Consistency Projection (PCP)
  3. Window Types:
     - 'hard'           : Legacy binary rectangular boxcar window (13 x 3, 13 x 5)
     - 'tukey_outward'  : Smooth outward raised-cosine tapering window (eliminates Gibbs ringing)
     - 'tukey_inward'   : Smooth inward raised-cosine tapering window
     - 'supergaussian'  : Smooth elliptical flat-top super-Gaussian window
  4. Auto-detection of Normalization (RMS, Standardization, MinMax) from model training logs.
  5. Publication-quality evaluation plots (.png + .pdf) and metric summaries.
========================================================================================
"""

import os
import sys
import re
import shutil
import time
import argparse
import numpy as np
import scipy.io as sio
import h5py
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import local FDA utilities & strategy library
from fda_window_utils import (
    create_fda_window,
    tf_to_delay_doppler,
    delay_doppler_to_tf,
    fda_blend_channels,
    compute_source_prototype_spectrum,
    apply_pilot_consistency_projection
)
from fda_strategies import (
    get_approach_executor,
    APPROACH_REGISTRY,
    predict_channel_onnx,
    rms_scaler,
    de_rms,
    standardize_scaler,
    de_standardize,
    minmax_scaler,
    de_min_max,
)

# ============================================================================
# DEFAULT CONFIGURATION CONSTANTS (For direct IDE execution)
# ============================================================================
MODEL_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_rms"
DATASET_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps"
OUT_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\inference_withFDA\A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz\LS_Attention_rms_dual_smooth13x3"
SOURCE_REF_PATH = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz"

APPROACH = "2.1"                # '2.1' (Dual FDA), '2.2', '2.3' (Raw), '2.4' (Post), '2.5' (Self-Refined), '3.1', '3.2', '3.4'
FDA_STAGE = "dual_pcp"          # Legacy fallback: 'none', 'pre', 'post', 'dual', 'dual_pcp'
WINDOW_TYPE = "tukey_outward"   # 'tukey_outward', 'hard', 'tukey_inward', 'supergaussian'
WIN_H = 13                      # Nominal window height along Delay
WIN_W = 3                       # Nominal window width along Doppler (3 or 5)
TAPER_H = 2                     # Delay transition rows (default 2 -> weights [0.75, 0.25])
TAPER_W = 1                     # Doppler transition cols (default 1 -> weight [0.50])
PCP_ALPHA = 0.5                 # Pilot consistency projection blending weight (0.0 to 1.0)
NUM_SAMPLES = None              # Integer (e.g. 64) or None for all
MODEL_NAME = "auto"             # 'auto', 'best_model.onnx', 'best_net.onnx', etc.
CLIP_EXTRAP = "auto"            # 'auto', 'true', 'false'
# ============================================================================


def find_onnx_model(sub_dir, preferred_name="auto"):
    """Adaptively finds an ONNX model file in sub_dir or sub_dir/results."""
    candidates = []
    if preferred_name and str(preferred_name).lower() not in ["auto", "none", ""]:
        candidates.append(preferred_name)
    default_candidates = ["best_model.onnx", "best_net.onnx", "final_model.onnx", "final_net.onnx"]
    for c in default_candidates:
        if c not in candidates:
            candidates.append(c)

    search_dirs = [os.path.join(sub_dir, "results"), sub_dir]
    for c in candidates:
        for s_dir in search_dirs:
            p = os.path.join(s_dir, c)
            if os.path.isfile(p):
                return p

    for s_dir in search_dirs:
        if os.path.isdir(s_dir):
            onnx_files = [f for f in os.listdir(s_dir) if f.lower().endswith('.onnx')]
            if onnx_files:
                onnx_files.sort(key=lambda x: (
                    0 if "best" in x.lower() else (1 if "final" in x.lower() else 2),
                    x.lower()
                ))
                return os.path.join(s_dir, onnx_files[0])
    return None


def extract_snr(folder_name):
    """Extracts integer SNR value from folder name (e.g., LS_-10, SNR_15dB)."""
    match = re.search(r'[-+]?\d+', folder_name)
    return int(match.group()) if match else None


def load_source_mat_file(mat_path):
    """Loads a MATLAB .mat file supporting both v7 (scipy.io) and v7.3 (h5py)."""
    try:
        data = sio.loadmat(mat_path)
        out = {}
        for k, v in data.items():
            if not k.startswith('__'):
                out[k] = v
        return out
    except Exception:
        with h5py.File(mat_path, 'r') as hf:
            out = {}
            for k in hf.keys():
                v = hf[k][()]
                if isinstance(v, np.ndarray) and v.dtype.names is not None:
                    if 'real' in v.dtype.names and 'imag' in v.dtype.names:
                        v = v['real'] + 1j * v['imag']
                out[k] = v
            return out


def align_channel_to_N_132_14(arr):
    """Standardizes channel dimensions to (N_samples, 132 subcarriers, 14 symbols)."""
    if arr.ndim == 2:
        if arr.shape == (132, 14):
            return arr[None, ...]
        elif arr.shape == (14, 132):
            return arr.T[None, ...]
        else:
            raise ValueError(f"Unexpected 2D channel shape: {arr.shape}")
    elif arr.ndim == 3:
        if arr.shape[1] == 132 and arr.shape[2] == 14:
            return arr
        elif arr.shape[0] == 14 and arr.shape[1] == 132:
            return np.transpose(arr, (2, 1, 0))
        elif arr.shape[0] == 132 and arr.shape[1] == 14:
            return np.transpose(arr, (2, 0, 1))
        elif arr.shape[1] == 14 and arr.shape[2] == 132:
            return np.transpose(arr, (0, 2, 1))
        else:
            raise ValueError(f"Unexpected 3D channel shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported array dimension: {arr.ndim}")


def match_original_shape(H_est_N_132_14, reference_mat_path):
    """Transposes H_est to match the exact dimension convention of the reference dataset."""
    try:
        f = load_source_mat_file(reference_mat_path)
        ref_shape = None
        for key in ['H_perfect', 'H_clean', 'H_li', 'H_LI', 'H_prac']:
            if key in f:
                ref_shape = f[key].shape
                break
        if ref_shape is not None:
            if ref_shape[0] == 14 and ref_shape[1] == 132:
                # (14, 132, N)
                return np.transpose(H_est_N_132_14, (2, 1, 0))
            elif ref_shape[0] == 132 and ref_shape[1] == 14:
                # (132, 14, N)
                return np.transpose(H_est_N_132_14, (1, 2, 0))
    except Exception:
        pass
    return H_est_N_132_14


# ============================================================================
# NORMALIZATION UTILITIES (Matching Training Preprocessing Exactly)
# ============================================================================

def minmax_scaler(x, lower_range=-1):
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1, 2)
    x_min = np.min(x_reshaped, axis=1)
    x_max = np.max(x_reshaped, axis=1)
    scale = np.clip(x_max - x_min, 1e-30, None)
    reshape_dims = [N] + [1] * (x.ndim - 2) + [2]
    x_min_b = x_min.reshape(reshape_dims)
    scale_b = scale.reshape(reshape_dims)
    x_scaled = (x - x_min_b) / scale_b
    if lower_range == -1:
        x_scaled = x_scaled * 2.0 - 1.0
    return x_scaled, x_min, x_max


def de_min_max(x_normd, x_min, x_max, lower_range=-1):
    N = x_normd.shape[0]
    reshape_dims = [N] + [1] * (x_normd.ndim - 2) + [2]
    x_min_b = x_min.reshape(reshape_dims)
    x_max_b = x_max.reshape(reshape_dims)
    if lower_range == -1:
        scale = (x_max_b - x_min_b) / 2.0
        shift = (x_max_b + x_min_b) / 2.0
        return x_normd * scale + shift
    else:
        scale = x_max_b - x_min_b
        return x_normd * scale + x_min_b


def rms_scaler(x):
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1, 2)
    x_rms = np.sqrt(np.mean(x_reshaped ** 2, axis=(1, 2)))  # Shape: (N,)
    x_rms = np.clip(x_rms, 1e-30, None)
    reshape_dims = [N] + [1] * (x.ndim - 1)
    x_rms_b = x_rms.reshape(reshape_dims)
    x_scaled = x / x_rms_b
    return x_scaled, x_rms


def de_rms(x_normd, x_rms):
    N = x_normd.shape[0]
    reshape_dims = [N] + [1] * (x_normd.ndim - 1)  # For (N, 132, 14, 2) -> (N, 1, 1, 1)
    x_rms_b = np.squeeze(x_rms).reshape(reshape_dims)
    return x_normd * x_rms_b


def standardize_scaler(x):
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1, 2)
    x_mean = np.mean(x_reshaped, axis=1)
    x_std = np.std(x_reshaped, axis=1)
    x_std = np.clip(x_std, 1e-30, None)
    reshape_dims = [N] + [1] * (x.ndim - 2) + [2]
    x_mean_b = x_mean.reshape(reshape_dims)
    x_std_b = x_std.reshape(reshape_dims)
    x_scaled = (x - x_mean_b) / x_std_b
    return x_scaled, x_mean, x_std


def de_standardize(x_normd, x_mean, x_std):
    N = x_normd.shape[0]
    reshape_dims = [N] + [1] * (x_normd.ndim - 2) + [2]
    x_mean_b = x_mean.reshape(reshape_dims)
    x_std_b = x_std.reshape(reshape_dims)
    return x_normd * x_std_b + x_mean_b


# ============================================================================
# METRICS & PLOTTING UTILITIES
# ============================================================================

def compute_mmse(h_pred, h_true):
    return float(np.mean(np.abs(h_pred - h_true) ** 2))


def compute_nmse(h_pred, h_true):
    err_power = np.sum(np.abs(h_pred - h_true) ** 2)
    ref_power = np.sum(np.abs(h_true) ** 2)
    return float(err_power / np.clip(ref_power, 1e-30, None))


def compute_nmse_db(h_pred, h_true):
    val = compute_nmse(h_pred, h_true)
    return float(10.0 * np.log10(np.clip(val, 1e-30, None)))


def compute_ssim_single(h_true_2d, h_pred_2d):
    """Calculates complex SSIM matching OpenNTN formulation."""
    def ssim_real(x, y):
        x = x.ravel().astype(np.float64)
        y = y.ravel().astype(np.float64)
        mu_x, mu_y = np.mean(x), np.mean(y)
        var_x, var_y = np.var(x), np.var(y)
        cov_xy = np.mean((x - mu_x) * (y - mu_y))
        L = max(float(np.max(x)), float(np.max(y))) - min(float(np.min(x)), float(np.min(y)))
        if L == 0:
            L = 1.0
        c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
        return float(((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) /
                     ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)))
    s_r = ssim_real(h_true_2d.real, h_pred_2d.real)
    s_i = ssim_real(h_true_2d.imag, h_pred_2d.imag)
    return (s_r + s_i) / 2.0


def compute_ssim_batch(h_pred, h_true):
    N = h_pred.shape[0]
    ssim_vals = [compute_ssim_single(h_true[i], h_pred[i]) for i in range(N)]
    return float(np.mean(ssim_vals))


def plot_comparison(H_true, H_pred, dest_folder, snr_val, fda_info_str=""):
    """Generates 4 representative side-by-side comparison plots (PNG + PDF)."""
    try:
        N = H_true.shape[0]
        sample_indices = np.linspace(0, N - 1, min(4, N), dtype=int)
        num_pairs = len(sample_indices)

        fig, axes = plt.subplots(num_pairs, 4, figsize=(16, 3.2 * num_pairs))
        if num_pairs == 1:
            axes = axes[None, :]

        for row_idx, s_idx in enumerate(sample_indices):
            ht = H_true[s_idx]
            hp = H_pred[s_idx]
            nmse_s = compute_nmse_db(hp[None, ...], ht[None, ...])

            # Real Part
            v_min_r = min(float(np.min(ht.real)), float(np.min(hp.real)))
            v_max_r = max(float(np.max(ht.real)), float(np.max(hp.real)))

            ax0 = axes[row_idx, 0]
            im0 = ax0.imshow(ht.real, aspect='auto', cmap='coolwarm', origin='lower', vmin=v_min_r, vmax=v_max_r)
            ax0.set_title(f"S#{s_idx} True (Real)", fontsize=9)
            ax0.set_ylabel(f"S#{s_idx}\nSubcarriers", fontsize=9)
            fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

            ax1 = axes[row_idx, 1]
            im1 = ax1.imshow(hp.real, aspect='auto', cmap='coolwarm', origin='lower', vmin=v_min_r, vmax=v_max_r)
            ax1.set_title(f"S#{s_idx} Inferred (Real) | {nmse_s:.1f} dB", fontsize=9)
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # Magnitude
            v_max_m = max(float(np.max(np.abs(ht))), float(np.max(np.abs(hp))))

            ax2 = axes[row_idx, 2]
            im2 = ax2.imshow(np.abs(ht), aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=v_max_m)
            ax2.set_title(f"S#{s_idx} True (|H|)", fontsize=9)
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            ax3 = axes[row_idx, 3]
            im3 = ax3.imshow(np.abs(hp), aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=v_max_m)
            ax3.set_title(f"S#{s_idx} Inferred (|H|) | {nmse_s:.1f} dB", fontsize=9)
            fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            if row_idx == num_pairs - 1:
                for ax in axes[row_idx]:
                    ax.set_xlabel("OFDM Symbols", fontsize=8)

        fig.suptitle(f"Inference Comparison @ {snr_val} dB {fda_info_str}", fontsize=12, fontweight='bold')
        fig.tight_layout()
        png_p = os.path.join(dest_folder, "comparison_real_and_magnitude.png")
        pdf_p = os.path.join(dest_folder, "comparison_real_and_magnitude.pdf")
        fig.savefig(png_p, dpi=200, bbox_inches='tight')
        fig.savefig(pdf_p, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"    [Plot] Saved comparison curves -> {png_p}")
    except Exception as err:
        print(f"    [Warning] Plot generation failed: {err}")


# ============================================================================
# MAIN INFERENCE ENGINE
# ============================================================================

def run_fda_inference(
    model_dir=MODEL_DIR,
    dataset_dir=DATASET_DIR,
    out_dir=OUT_DIR,
    source_ref_path=SOURCE_REF_PATH,
    approach=APPROACH,
    fda_stage=FDA_STAGE,
    window_type=WINDOW_TYPE,
    win_h=WIN_H,
    win_w=WIN_W,
    taper_h=TAPER_H,
    taper_w=TAPER_W,
    pcp_alpha=PCP_ALPHA,
    num_samples=NUM_SAMPLES,
    model_name=MODEL_NAME,
    clip_extrap=CLIP_EXTRAP,
    model_type="auto",
    model2_dir=None
):
    # Resolve Approach: support backward compatibility with fda_stage
    approach = str(approach).lower().strip()
    if fda_stage.lower() == "none" and approach in ["2.1", "dual", "dual_pcp"]:
        approach = "2.3"  # Raw baseline
    elif fda_stage.lower() == "post" and approach in ["2.1", "dual", "dual_pcp"]:
        approach = "2.4"  # Post-FDA only

    print("\n" + "="*80)
    print("  ONNX Cross-Domain Inference with Test-Time FDA & Smooth Windowing")
    print("="*80)
    print(f"Model Directory       : {model_dir}")
    print(f"Dataset Directory     : {dataset_dir}")
    print(f"Output Directory      : {out_dir}")
    print(f"Source Reference Path : {source_ref_path}")
    print(f"FDA Approach Strategy : Approach {approach.upper()}")
    print(f"FDA Window Type       : {window_type} (Nominal: {win_h}x{win_w}, Taper: {taper_h}x{taper_w})")
    print(f"Pilot Consistency (PCP): Alpha = {pcp_alpha} ({'Enabled' if pcp_alpha > 0 else 'Disabled'})")
    print(f"Num Samples Limit     : {num_samples if num_samples is not None else 'All'}")
    print("="*80 + "\n")

    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' does not exist.")
        sys.exit(1)
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # 1. Determine Model Architecture Type (Sequence vs Grid)
    is_sequence_model = False
    model_name_lower = os.path.basename(model_dir).lower()
    if "ls_attention" in model_name_lower or "ls_transformer" in model_name_lower or "lssequence" in model_name_lower:
        is_sequence_model = True
    elif "li_dncnn" in model_name_lower or "dncnn" in model_name_lower or "cgan" in model_name_lower:
        is_sequence_model = False
    else:
        # Check subfolders or ONNX input shape
        if str(model_type).lower() == "ls":
            is_sequence_model = True
        elif str(model_type).lower() == "li":
            is_sequence_model = False

    print(f"[Architecture] Detected Model Type: {'Sequence-based (Pilot Tokens)' if is_sequence_model else 'Grid-based (2D Interpolation)'}")

    # 2. Determine Normalization Scheme
    norm_mode = "rms"
    for root, _, files in os.walk(model_dir):
        if "final_epoch.txt" in files:
            fe_path = os.path.join(root, "final_epoch.txt")
            try:
                with open(fe_path, "r") as fe_f:
                    fe_content = fe_f.read().lower()
                    if re.search(r'rms\s*[:=]\s*true', fe_content):
                        norm_mode = "rms"
                    elif re.search(r'standardize\s*[:=]\s*true', fe_content):
                        norm_mode = "standardize"
                    elif re.search(r'standardize\s*[:=]\s*false', fe_content) and re.search(r'rms\s*[:=]\s*false', fe_content):
                        norm_mode = "minmax"
                print(f"[Auto-detect] Normalization Mode: '{norm_mode.upper()}' (from '{fe_path}')")
                break
            except Exception:
                pass
    if norm_mode is None:
        if "standardize" in model_name_lower or "_std" in model_name_lower:
            norm_mode = "standardize"
        elif "rms" in model_name_lower:
            norm_mode = "rms"
        else:
            norm_mode = "minmax"
        print(f"[Auto-detect] Normalization Mode: '{norm_mode.upper()}' (from folder name)")

    # 3. Precompute Source Reference Spectrum if strategy requires it
    need_source_proto = approach in [
        "2.1", "2.2", "2.5", "3.3",
        "app2_1", "app2_2", "app2_5", "app3_3",
        "dual", "dual_pcp", "model2_dual"
    ]
    source_amp_prototype = None
    if need_source_proto:
        if not source_ref_path or not os.path.exists(source_ref_path):
            print(f"[Warning] Source reference path '{source_ref_path}' not found! Attempting fallback discovery...")
            ws_root = os.path.abspath(os.path.join(model_dir, "..", ".."))
            cand = os.path.join(ws_root, "generatedChan")
            if os.path.exists(cand):
                for d in os.listdir(cand):
                    if "a100" in d.lower():
                        source_ref_path = os.path.join(cand, d)
                        break
        if source_ref_path and os.path.exists(source_ref_path):
            print(f"--> Precomputing Source Reference Delay-Doppler Spectrum from: {source_ref_path} ...")
            source_amp_prototype = compute_source_prototype_spectrum(source_ref_path, max_samples=256)
            print(f"    Source Prototype Spectrum computed successfully (shape: {source_amp_prototype.shape})")
        else:
            print("[Warning] Could not resolve source reference spectrum. Pre-FDA will use target amplitude baseline.")

    # 4. Find all SNR folders in dataset directory
    target_subs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    target_subs.sort(key=lambda x: extract_snr(x) if extract_snr(x) is not None else 999)

    results_summary = []

    # Iterate over each SNR subfolder
    for sub in target_subs:
        snr = extract_snr(sub)
        if snr is None:
            continue

        snr_folder_path = os.path.join(dataset_dir, sub)
        # Search for dataset file
        mat_candidates = ["matlabNTN.mat", "testChannel.mat", "testChannel_target.mat", f"matlabNTN_{snr}dB.mat"]
        source_mat_path = None
        for mc in mat_candidates:
            p = os.path.join(snr_folder_path, mc)
            if os.path.isfile(p):
                source_mat_path = p
                break
        if not source_mat_path:
            # Fallback to any .mat file
            mats = [f for f in os.listdir(snr_folder_path) if f.lower().endswith('.mat') and not f.startswith("inferred")]
            if mats:
                source_mat_path = os.path.join(snr_folder_path, mats[0])

        if not source_mat_path:
            continue

        # Find corresponding ONNX model for this SNR
        # Check subfolders of model_dir matching SNR
        onnx_path = None
        for m_sub in os.listdir(model_dir):
            if os.path.isdir(os.path.join(model_dir, m_sub)):
                if extract_snr(m_sub) == snr:
                    onnx_path = find_onnx_model(os.path.join(model_dir, m_sub), preferred_name=model_name)
                    if onnx_path:
                        break
        # Fallback to root model_dir
        if not onnx_path:
            onnx_path = find_onnx_model(model_dir, preferred_name=model_name)

        if not onnx_path:
            print(f"Skipping SNR {snr} dB: No ONNX model found.")
            continue

        prefix = "LS" if is_sequence_model else "LI"
        dest_snr_folder = os.path.join(out_dir, f"{prefix}_{snr}dB")
        os.makedirs(dest_snr_folder, exist_ok=True)
        dest_mat_path = os.path.join(dest_snr_folder, "inferredChannel.mat")

        print(f"\n--> Processing SNR {snr} dB:")
        print(f"  Target File : {source_mat_path}")
        print(f"  ONNX Model  : {onnx_path}")
        print(f"  Destination : {dest_mat_path}")

        # Load Dataset
        try:
            f = load_source_mat_file(source_mat_path)
            # Perfect Channel
            H_perf = None
            for pk in ['H_perfect', 'H_clean', 'H_true', 'H_perf']:
                if pk in f:
                    H_perf = align_channel_to_N_132_14(f[pk])
                    break
            if H_perf is None:
                print(f"  [Error] No perfect channel found in {source_mat_path}. Skipping.")
                continue

            # Pilot coordinates
            pilot_rows = np.squeeze(f.get('pilot_rows', np.array([]))).astype(int)
            pilot_cols = np.squeeze(f.get('pilot_cols', np.array([]))).astype(int)
            if pilot_rows.size > 0 and np.min(pilot_rows) >= 1:
                pilot_rows = pilot_rows - 1  # 0-based
            if pilot_cols.size > 0 and np.min(pilot_cols) >= 1:
                pilot_cols = pilot_cols - 1

            # Input Channel
            H_li = None
            for lk in ['H_li', 'H_LI', 'H_linear', 'H_prac', 'H_practical']:
                if lk in f:
                    H_li = align_channel_to_N_132_14(f[lk])
                    break

            H_pilots = None
            for lsk in ['H_ls_pilots', 'H_ls_pilots_ori', 'H_LS_comp', 'H_ls']:
                if lsk in f:
                    H_pilots = f[lsk]
                    break

            if H_pilots is not None:
                if pilot_rows.size > 0 and H_pilots.shape[0] == len(pilot_rows) and H_pilots.shape[1] != len(pilot_rows):
                    H_pilots = H_pilots.T
                elif H_pilots.shape[0] != H_perf.shape[0] and H_pilots.shape[1] == H_perf.shape[0]:
                    H_pilots = H_pilots.T

            if num_samples is not None:
                H_perf = H_perf[:num_samples]
                if H_li is not None:
                    H_li = H_li[:num_samples]
                if H_pilots is not None:
                    H_pilots = H_pilots[:num_samples]

            N_samples = H_perf.shape[0]
            print(f"  Loaded {N_samples} samples. Perfect shape: {H_perf.shape}")

        except Exception as load_err:
            print(f"  [Error] Failed to load dataset: {load_err}")
            continue

        start_time = time.perf_counter()

        # --------------------------------------------------------------------
        # Prepare Target Grid Channel
        # --------------------------------------------------------------------
        if H_li is not None:
            H_target_grid = np.copy(H_li)
        elif H_pilots is not None and pilot_rows.size > 0:
            H_target_grid = np.zeros((N_samples, 132, 14), dtype=np.complex128)
            for i in range(N_samples):
                H_target_grid[i, pilot_rows, pilot_cols] = H_pilots[i, :]
        else:
            H_target_grid = np.copy(H_perf)

        # --------------------------------------------------------------------
        # Strategy Parameters Setup
        # --------------------------------------------------------------------
        window_kwargs = {
            "win_h_px": win_h,
            "win_w_px": win_w,
            "window_type": window_type,
            "taper_h": taper_h,
            "taper_w": taper_w,
            "scale_mode": norm_mode,
        }
        model_cfg = {
            "is_sequence_model": is_sequence_model,
            "model_name_lower": model_name_lower,
            "norm_mode": norm_mode,
            "pilot_rows": pilot_rows,
            "pilot_cols": pilot_cols,
        }

        # Find model2 ONNX if specified (for Approach 3.4)
        onnx_model2_path = None
        if model2_dir:
            for m_sub in os.listdir(model2_dir):
                if os.path.isdir(os.path.join(model2_dir, m_sub)) and extract_snr(m_sub) == snr:
                    onnx_model2_path = find_onnx_model(os.path.join(model2_dir, m_sub), preferred_name=model_name)
                    if onnx_model2_path:
                        break
            if not onnx_model2_path:
                onnx_model2_path = find_onnx_model(model2_dir, preferred_name=model_name)

        # --------------------------------------------------------------------
        # Execute Selected FDA Strategy
        # --------------------------------------------------------------------
        try:
            strategy_fn = get_approach_executor(approach)
            print(f"  [Approach {approach}] Running {strategy_fn.__name__} via {window_type} ({win_h}x{win_w})...")
            H_final = strategy_fn(
                H_target=H_target_grid,
                onnx_path=onnx_path,
                source_ref=source_amp_prototype,
                window_kwargs=window_kwargs,
                model_cfg=model_cfg,
                onnx_model2_path=onnx_model2_path,
                alpha=pcp_alpha,
            )
        except Exception as exec_err:
            print(f"  [Error] Strategy execution failed: {exec_err}")
            continue

        elapsed = time.perf_counter() - start_time
        avg_time = elapsed / N_samples

        # --------------------------------------------------------------------
        # Step 7: Calculate Metrics
        # --------------------------------------------------------------------
        mmse = compute_mmse(H_final, H_perf)
        nmse = compute_nmse(H_final, H_perf)
        nmse_db = compute_nmse_db(H_final, H_perf)
        try:
            ssim = compute_ssim_batch(H_final, H_perf)
        except Exception:
            ssim = float('nan')

        print(f"  [Metrics] NMSE: {nmse:.6f} | NMSE (dB): {nmse_db:.2f} dB | MMSE: {mmse:.6e} | SSIM: {ssim:.6f}")

        # --------------------------------------------------------------------
        # Step 8: Save Inferred Channel MAT file
        # --------------------------------------------------------------------
        H_infer_save = match_original_shape(H_final, source_mat_path)
        save_dict = {
            'H_infer': H_infer_save,
            'H_perfect': f.get('H_perfect', match_original_shape(H_perf, source_mat_path)),
            'SNRdB': snr,
            'pilot_rows': pilot_rows + 1,  # 1-based for MATLAB
            'pilot_cols': pilot_cols + 1,
        }
        if H_li is not None:
            save_dict['H_li'] = match_original_shape(H_li, source_mat_path)
        if H_pilots is not None:
            save_dict['H_ls_pilots'] = H_pilots

        sio.savemat(dest_mat_path, save_dict)
        print(f"  Saved -> {dest_mat_path}")

        # --------------------------------------------------------------------
        # Step 9: Save Visualizations
        # --------------------------------------------------------------------
        fda_label = f"[{fda_stage.upper()} | {window_type} {win_h}x{win_w}]"
        plot_comparison(H_perf, H_final, dest_snr_folder, snr, fda_info_str=fda_label)

        results_summary.append({
            'snr': snr,
            'nmse': nmse,
            'nmse_db': nmse_db,
            'mmse': mmse,
            'ssim': ssim,
            'avg_time': avg_time
        })

    # Write summary info.md
    info_md_path = os.path.join(out_dir, "info.md")
    with open(info_md_path, "w") as imf:
        imf.write(f"# FDA Test-Time Inference Summary\n\n")
        imf.write(f"- **FDA Approach**: `Approach {approach.upper()}`\n")
        imf.write(f"- **FDA Stage Legacy**: `{fda_stage}`\n")
        imf.write(f"- **Window Type**: `{window_type}`\n")
        imf.write(f"- **Window Geometry**: Delay = {win_h} px, Doppler = {win_w} px\n")
        imf.write(f"- **Taper Setting**: Delay taper = {taper_h} rows, Doppler taper = {taper_w} cols\n")
        imf.write(f"- **Pilot Consistency**: Alpha = {pcp_alpha}\n")
        imf.write(f"- **Model Directory**: `{model_dir}`\n")
        if model2_dir:
            imf.write(f"- **Model 2 Directory**: `{model2_dir}`\n")
        imf.write(f"- **Dataset Directory**: `{dataset_dir}`\n")
        imf.write(f"- **Normalization**: `{norm_mode}`\n\n")
        imf.write("## Evaluation Metrics\n\n")
        imf.write("| SNR (dB) | NMSE | NMSE (dB) | MMSE | SSIM | Avg Time/Sample (s) |\n")
        imf.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results_summary:
            imf.write(f"| {r['snr']} | {r['nmse']:.6f} | {r['nmse_db']:.2f} | {r['mmse']:.6e} | {r['ssim']:.4f} | {r['avg_time']:.4f} |\n")

    print(f"\n[Done] Summary saved to: {info_md_path}")
    print("="*80 + "\n")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test-Time FDA ONNX Channel Inference Pipeline")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR, help="Path to trained model directory")
    parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR, help="Path to target dataset directory")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR, help="Path to output directory")
    parser.add_argument("--source-ref-path", type=str, default=SOURCE_REF_PATH, help="Path to source dataset for reference spectrum")
    parser.add_argument("--approach", type=str, default="2.1",
                        help="FDA Approach Strategy: 2.1 (Dual FDA), 2.2 (Clean Source Center), 2.3 (Raw Baseline), "
                             "2.4 (Post-FDA Only), 2.5 (Self-Refined 2-Pass), 2.6/3.4 (Hybrid Ensemble), "
                             "3.1 (Model 2 Raw), 3.2 (Model 2 Post-FDA), 3.3 (Model 2 Dual Alignment)")
    parser.add_argument("--model2-dir", type=str, default=None,
                        help="Path to trained model2 directory for Approach 3.4 ensemble")
    parser.add_argument("--fda-stage", type=str, default=FDA_STAGE, choices=["none", "pre", "post", "dual", "dual_pcp"],
                        help="Legacy FDA Adaptation stage: 'none', 'pre', 'post', 'dual', 'dual_pcp'")
    parser.add_argument("--window-type", type=str, default=WINDOW_TYPE,
                        choices=["hard", "tukey_outward", "tukey_inward", "supergaussian"],
                        help="Window mask type: 'hard' (legacy binary) or 'tukey_outward' (smooth outward taper)")
    parser.add_argument("--win-h", type=int, default=WIN_H, help="Window height along Delay (default 13)")
    parser.add_argument("--win-w", type=int, default=WIN_W, help="Window width along Doppler (default 3 or 5)")
    parser.add_argument("--taper-h", type=int, default=TAPER_H, help="Delay transition rows (default 2)")
    parser.add_argument("--taper-w", type=int, default=TAPER_W, help="Doppler transition cols (default 1)")
    parser.add_argument("--pcp-alpha", type=float, default=PCP_ALPHA, help="Pilot consistency projection alpha [0.0 to 1.0]")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="Number of samples to evaluate (default all)")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Preferred ONNX model filename")
    parser.add_argument("--clip-extrap", type=str, default=CLIP_EXTRAP, help="Extrapolation clipping: 'auto', 'true', 'false'")
    parser.add_argument("--model-type", type=str, default="auto", choices=["auto", "LS", "LI"], help="Model type override")
    # Clean up sys.argv in case compound arguments with spaces were passed (e.g. from PowerShell)
    cleaned_argv = []
    for arg in sys.argv[1:]:
        if arg.startswith("--") and " " in arg:
            parts = arg.split(" ", 1)
            cleaned_argv.extend(parts)
        else:
            cleaned_argv.append(arg)

    args = parser.parse_args(cleaned_argv)

    run_fda_inference(
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        source_ref_path=args.source_ref_path,
        approach=args.approach,
        fda_stage=args.fda_stage,
        window_type=args.window_type,
        win_h=args.win_h,
        win_w=args.win_w,
        taper_h=args.taper_h,
        taper_w=args.taper_w,
        pcp_alpha=args.pcp_alpha,
        num_samples=args.num_samples,
        model_name=args.model_name,
        clip_extrap=args.clip_extrap,
        model_type=args.model_type,
        model2_dir=args.model2_dir
    )
