"""
Approach B: Test-Time Domain Adaptation via Physical Coordinate Warping
Zero Retraining / Real-Time Inference:
1. Inverse-warps target pilots (compresses 300ns -> 100ns).
2. Passes disguised pilots through frozen Source Model 1 (ONNX).
3. Forward-warps predicted grid back to target domain (stretches 100ns -> 300ns).
4. Anchors measured target pilots to ensure zero pilot error.
5. Evaluates NMSE and SSIM against baseline straight inference.
"""

import os
import sys
import argparse
import time
import re
import numpy as np
import scipy.io
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.io_utils import load_channel_mat, save_channel_mat, get_default_dmrs_coords
from core.warp_operators import (
    warp_source_to_target,
    inverse_warp_target_to_source,
    warp_pilots_vector,
    anchor_pilots_into_grid,
    dilate_delay_channel_grid,
    modulate_doppler_channel_grid
)
from core.estimate_pilot_shifts import estimate_delay_dilation_autocorr, estimate_bulk_doppler_shift
from core.metrics import compute_nmse, compute_nmse_db, compute_ssim_batch


def find_onnx_model(sub_dir):
    """Finds ONNX model in sub_dir or sub_dir/results."""
    candidates = ["best_net.onnx", "best_model.onnx", "final_model.onnx", "final_net.onnx"]
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
                onnx_files.sort(key=lambda x: (0 if "best" in x.lower() else 1, x.lower()))
                return os.path.join(s_dir, onnx_files[0])
    return None


def run_test_time_warping(
    model_dir,
    target_dataset_dir,
    source_ref_dir=None,
    out_dir=r"math_translation\results\approachB_test_time",
    beta_tau=None,
    delta_nu=None,
    num_samples=None,
    pilot_anchor_blend=1.0
):
    print("\n" + "=" * 80)
    print("  Approach B: Test-Time Adaptation via Physical Coordinate Warping (Zero Retraining)")
    print("=" * 80)
    print(f"Model Directory       : {model_dir}")
    print(f"Target Dataset Dir    : {target_dataset_dir}")
    print(f"Output Directory      : {out_dir}")
    print(f"Pilot Anchoring Blend : {pilot_anchor_blend}")
    print(f"Num Samples           : {num_samples if num_samples is not None else 'All'}")
    print("=" * 80 + "\n")

    os.makedirs(out_dir, exist_ok=True)

    # Detect normalization mode from model path
    model_dir_lower = model_dir.lower()
    if "rms" in model_dir_lower:
        norm_mode = "rms"
    elif "standardize" in model_dir_lower or "_std" in model_dir_lower:
        norm_mode = "standardize"
    else:
        norm_mode = "minmax"
    print(f"[*] Detected Model Normalization Mode: {norm_mode.upper()}")

    # Find model SNR subfolders
    model_subs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    def extract_snr(s):
        m = re.search(r'[-+]?\d+', s)
        return int(m.group()) if m else None

    model_subs = sorted(model_subs, key=lambda x: extract_snr(x) if extract_snr(x) is not None else 999)

    # Find target SNR subfolders
    target_subs = [d for d in os.listdir(target_dataset_dir) if os.path.isdir(os.path.join(target_dataset_dir, d))]

    # Estimate physical shifts blindly if not given
    if beta_tau is None or delta_nu is None:
        print("[*] Performing blind estimation of physical shifts from target pilots...")
        # Find reference 10dB or first available
        tgt_ref_sub = next((d for d in target_subs if extract_snr(d) == 10), target_subs[0])
        _, p_tgt_ref, r_p, c_p, _ = load_channel_mat(os.path.join(target_dataset_dir, tgt_ref_sub), 'ls')

        if source_ref_dir and os.path.exists(source_ref_dir):
            src_ref_sub = next((d for d in os.listdir(source_ref_dir) if extract_snr(d) == 10), None)
            src_p = os.path.join(source_ref_dir, src_ref_sub) if src_ref_sub else source_ref_dir
            _, p_src_ref, _, _, _ = load_channel_mat(src_p, 'ls')
        else:
            # Fallback to default MATLAB source A100 directory if present
            default_src = r"generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz\SNR_10dB"
            if os.path.exists(default_src):
                _, p_src_ref, _, _, _ = load_channel_mat(default_src, 'ls')
            else:
                p_src_ref = None

        if p_src_ref is not None:
            b_est, s_info, t_info = estimate_delay_dilation_autocorr(p_src_ref, p_tgt_ref, r_p, c_p)
            dnu_est, _, _ = estimate_bulk_doppler_shift(p_src_ref, p_tgt_ref, r_p, c_p)
            if beta_tau is None:
                beta_tau = b_est
            if delta_nu is None:
                delta_nu = dnu_est
            print(f"  [Blind Estimation] Delay Dilation Beta_tau: {beta_tau:.3f}, Doppler Shift: {delta_nu:.2f} Hz")
        else:
            if beta_tau is None:
                beta_tau = 3.0  # Default physical ratio A100 -> DUR300
            if delta_nu is None:
                delta_nu = 0.0
            print(f"  [Default Settings] Beta_tau: {beta_tau:.3f}, Delta_nu: {delta_nu:.2f} Hz")
    else:
        print(f"  [Provided Settings] Beta_tau: {beta_tau:.3f}, Delta_nu: {delta_nu:.2f} Hz")

    comparison_results = []

    for m_sub in model_subs:
        snr = extract_snr(m_sub)
        if snr is None:
            continue

        onnx_file = find_onnx_model(os.path.join(model_dir, m_sub))
        if onnx_file is None:
            continue

        tgt_sub = next((d for d in target_subs if extract_snr(d) == snr), None)
        if tgt_sub is None:
            continue

        tgt_path = os.path.join(target_dataset_dir, tgt_sub)
        print(f"\n>>> Evaluating SNR = {snr} dB (Model: {m_sub})")

        H_tgt_true, p_tgt, r_p, c_p, mat_dict = load_channel_mat(tgt_path, 'ls')
        if num_samples is not None:
            H_tgt_true = H_tgt_true[:num_samples]
            p_tgt = p_tgt[:num_samples]

        N_eval = p_tgt.shape[0]

        # -------------------------------------------------------------
        # 1. Straight Inference Baseline (No Warping)
        # -------------------------------------------------------------
        sess = ort.InferenceSession(onnx_file)
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        # Prepare normalized inputs for baseline
        x_raw = np.stack([p_tgt.real, p_tgt.imag], axis=-1).astype(np.float32)
        if norm_mode == "rms":
            p_rms = np.sqrt(np.mean(np.sum(x_raw**2, axis=-1), axis=1, keepdims=True)) + 1e-30
            x_base_in = x_raw / p_rms[..., np.newaxis]
        elif norm_mode == "standardize":
            b_mean = np.mean(x_raw, axis=1, keepdims=True)
            b_std = np.std(x_raw, axis=1, keepdims=True) + 1e-30
            x_base_in = (x_raw - b_mean) / b_std
        else:
            x_min = np.min(x_raw, axis=1, keepdims=True)
            x_max = np.max(x_raw, axis=1, keepdims=True)
            x_base_in = (x_raw - x_min) / (x_max - x_min + 1e-30) * 2.0 - 1.0

        # Run straight inference
        y_base_list = []
        for i in range(N_eval):
            y_out = sess.run([out_name], {in_name: x_base_in[i:i+1]})[0]
            y_base_list.append(y_out[0])
        y_base = np.array(y_base_list)

        # Denormalize straight inference
        if norm_mode == "rms":
            y_base_denorm = y_base * p_rms.reshape(N_eval, 1, 1, 1)
        elif norm_mode == "standardize":
            y_base_denorm = y_base * b_std.reshape(N_eval, 1, 1, 2) + b_mean.reshape(N_eval, 1, 1, 2)
        else:
            scale_min = x_min.reshape(N_eval, 1, 1, 2)
            scale_max = x_max.reshape(N_eval, 1, 1, 2)
            y_base_denorm = (y_base + 1.0) / 2.0 * (scale_max - scale_min) + scale_min
        H_straight = y_base_denorm[..., 0] + 1j * y_base_denorm[..., 1]

        nmse_straight_db = compute_nmse_db(H_straight, H_tgt_true)
        ssim_straight = compute_ssim_batch(H_straight, H_tgt_true)

        # -------------------------------------------------------------
        # 2. Approach B: Test-Time Warping Pipeline
        # -------------------------------------------------------------
        t_start = time.perf_counter()

        # Step 2a: Inverse-warp target pilots: compress 300ns -> 100ns
        p_tgt_disguised = warp_pilots_vector(
            p_tgt, r_p, c_p, beta_tau=beta_tau, delta_nu=delta_nu, forward=False
        )

        # Step 2b: Normalize disguised pilots
        x_disg = np.stack([p_tgt_disguised.real, p_tgt_disguised.imag], axis=-1).astype(np.float32)
        if norm_mode == "rms":
            d_rms = np.sqrt(np.mean(np.sum(x_disg**2, axis=-1), axis=1, keepdims=True)) + 1e-30
            x_disg_in = x_disg / d_rms[..., np.newaxis]
        elif norm_mode == "standardize":
            d_mean = np.mean(x_disg, axis=1, keepdims=True)
            d_std = np.std(x_disg, axis=1, keepdims=True) + 1e-30
            x_disg_in = (x_disg - d_mean) / d_std
        else:
            d_min = np.min(x_disg, axis=1, keepdims=True)
            d_max = np.max(x_disg, axis=1, keepdims=True)
            x_disg_in = (x_disg - d_min) / (d_max - d_min + 1e-30) * 2.0 - 1.0

        # Step 2c: Forward pass through frozen Model 1
        y_warped_list = []
        for i in range(N_eval):
            y_out = sess.run([out_name], {in_name: x_disg_in[i:i+1]})[0]
            y_warped_list.append(y_out[0])
        y_warped_pred = np.array(y_warped_list)

        # Step 2d: Denormalize in disguised space
        if norm_mode == "rms":
            y_pred_denorm = y_warped_pred * d_rms.reshape(N_eval, 1, 1, 1)
        elif norm_mode == "standardize":
            y_pred_denorm = y_warped_pred * d_std.reshape(N_eval, 1, 1, 2) + d_mean.reshape(N_eval, 1, 1, 2)
        else:
            d_scale_min = d_min.reshape(N_eval, 1, 1, 2)
            d_scale_max = d_max.reshape(N_eval, 1, 1, 2)
            y_pred_denorm = (y_warped_pred + 1.0) / 2.0 * (d_scale_max - d_scale_min) + d_scale_min
        H_pred_source_domain = y_pred_denorm[..., 0] + 1j * y_pred_denorm[..., 1]

        # Step 2e: Forward-warp predicted grid: stretch 100ns -> 300ns & add Doppler
        H_warped_back = warp_source_to_target(H_pred_source_domain, beta_tau=beta_tau, delta_nu=delta_nu)

        # Step 2f: Pilot Anchoring (ensure exact pilot fidelity)
        if pilot_anchor_blend > 0:
            H_adapted_final = anchor_pilots_into_grid(H_warped_back, p_tgt, r_p, c_p, blend_factor=pilot_anchor_blend)
        else:
            H_adapted_final = H_warped_back

        t_elapsed = time.perf_counter() - t_start

        nmse_adapted_db = compute_nmse_db(H_adapted_final, H_tgt_true)
        ssim_adapted = compute_ssim_batch(H_adapted_final, H_tgt_true)

        gain_db = nmse_straight_db - nmse_adapted_db
        print(f"  [Straight Infer] NMSE: {nmse_straight_db:6.2f} dB | SSIM: {ssim_straight:.4f}")
        print(f"  [Approach B]     NMSE: {nmse_adapted_db:6.2f} dB | SSIM: {ssim_adapted:.4f} | Gain: +{gain_db:.2f} dB (Time: {t_elapsed:.2f}s)")

        # Save inferred channel MAT file
        dest_snr_dir = os.path.join(out_dir, f"LS_{snr}dB")
        os.makedirs(dest_snr_dir, exist_ok=True)
        dest_mat = os.path.join(dest_snr_dir, "inferredChannel.mat")

        save_dict = {
            'H_LS_infer': np.transpose(H_adapted_final, (2, 1, 0)),
            'H_straight_infer': np.transpose(H_straight, (2, 1, 0)),
            'nmse_db': nmse_adapted_db,
            'ssim': ssim_adapted,
            'nmse_straight_db': nmse_straight_db,
            'beta_tau': beta_tau,
            'delta_nu': delta_nu
        }
        scipy.io.savemat(dest_mat, save_dict)

        comparison_results.append({
            'snr': snr,
            'nmse_straight': nmse_straight_db,
            'ssim_straight': ssim_straight,
            'nmse_adapted': nmse_adapted_db,
            'ssim_adapted': ssim_adapted,
            'gain_db': gain_db
        })

    # Generate summary plots and note
    plot_path = os.path.join(out_dir, "approachB_snr_curves.png")
    plot_approachB_results(comparison_results, plot_path)

    note_path = os.path.join(out_dir, "results_summary.md")
    with open(note_path, "w") as fh:
        fh.write("# Approach B: Test-Time Domain Adaptation Benchmark\n\n")
        fh.write(f"- **Model:** `{os.path.basename(model_dir)}`\n")
        fh.write(f"- **Target Dataset:** `{os.path.basename(target_dataset_dir)}`\n")
        fh.write(f"- **Beta_tau:** `{beta_tau:.3f}` | **Delta_nu:** `{delta_nu:.2f} Hz`\n")
        fh.write(f"- **Pilot Anchoring:** `{pilot_anchor_blend * 100:.0f}%`\n\n")
        fh.write("| SNR (dB) | Straight Infer NMSE (dB) | Approach B NMSE (dB) | Gain (dB) | Straight SSIM | Approach B SSIM |\n")
        fh.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in comparison_results:
            fh.write(f"| {r['snr']} | {r['nmse_straight']:.2f} | {r['nmse_adapted']:.2f} | **+{r['gain_db']:.2f}** | {r['ssim_straight']:.4f} | {r['ssim_adapted']:.4f} |\n")
        fh.write(f"\n![Comparison Curves](./approachB_snr_curves.png)\n")

    print(f"\n[DONE] Approach B evaluation completed! Results saved to: {out_dir}\n")


def plot_approachB_results(records, save_path):
    try:
        snrs = [r['snr'] for r in records]
        straight = [r['nmse_straight'] for r in records]
        adapted = [r['nmse_adapted'] for r in records]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(snrs, straight, 'o--', color='crimson', label='Straight Inference (Baseline)')
        ax.plot(snrs, adapted, 's-', color='teal', linewidth=2, label='Approach B: Test-Time Warping (Zero Retraining)')
        ax.set_xlabel("Channel SNR (dB)", fontsize=11)
        ax.set_ylabel("NMSE (dB)", fontsize=11)
        ax.set_title("Test-Time Physical Warping vs Straight Inference", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)

        for s, v_st, v_ad in zip(snrs, straight, adapted):
            gain = v_st - v_ad
            ax.annotate(f"+{gain:.1f} dB", xy=(s, v_ad), xytext=(s, v_ad - 1.5),
                        ha='center', fontsize=9, color='darkgreen', fontweight='bold')

        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"  [Warning] Failed to save curve plot: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Approach B: Test-Time Physical Coordinate Warping Inference")
    parser.add_argument("--model-dir", type=str, default=r"single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_rms")
    parser.add_argument("--target-dataset-dir", type=str, default=r"generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps")
    parser.add_argument("--source-ref-dir", type=str, default=r"generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz")
    parser.add_argument("--out-dir", type=str, default=r"math_translation\results\approachB_test_time")
    parser.add_argument("--beta-tau", type=float, default=None)
    parser.add_argument("--delta-nu", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--pilot-anchor-blend", type=float, default=1.0)

    args = parser.parse_args()
    run_test_time_warping(
        model_dir=args.model_dir,
        target_dataset_dir=args.target_dataset_dir,
        source_ref_dir=args.source_ref_dir,
        out_dir=args.out_dir,
        beta_tau=args.beta_tau,
        delta_nu=args.delta_nu,
        num_samples=args.num_samples,
        pilot_anchor_blend=args.pilot_anchor_blend
    )
