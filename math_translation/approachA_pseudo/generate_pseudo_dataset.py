"""
Approach A: Physical Pseudo-Dataset Generation
Translates source channel ground-truth labels into realistic target pseudo ground-truth
using blind physical Delay Dilation (beta_tau) and Doppler phase modulation (delta_nu).
Simulates 5G NR transmission, AWGN noise, LS pilot extraction, and 2D interpolation.
"""

import os
import sys
import argparse
import time
import numpy as np
import scipy.io
import scipy.interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure math_translation root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.io_utils import load_channel_mat, save_channel_mat, get_default_dmrs_coords
from core.warp_operators import warp_source_to_target
from core.estimate_pilot_shifts import estimate_delay_dilation_autocorr, estimate_bulk_doppler_shift
from core.metrics import compute_nmse, compute_nmse_db, compute_ssim_batch


def interpolate_pilots_2d(H_pilots, pilot_rows, pilot_cols, n_sc=132, n_sym=14):
    """
    Performs 2D scattered linear interpolation of pilot observations across the full grid.
    H_pilots: (N, 88)
    Returns H_li: (N, n_sc, n_sym)
    """
    N = H_pilots.shape[0]
    grid_r, grid_c = np.meshgrid(np.arange(n_sc), np.arange(n_sym), indexing='ij')
    points = np.column_stack([pilot_rows, pilot_cols])

    H_li = np.zeros((N, n_sc, n_sym), dtype=np.complex128)

    for n in range(N):
        real_p = H_pilots[n].real
        imag_p = H_pilots[n].imag

        # 2D Linear interpolation
        real_grid = scipy.interpolate.griddata(points, real_p, (grid_r, grid_c), method='linear')
        imag_grid = scipy.interpolate.griddata(points, imag_p, (grid_r, grid_c), method='linear')

        # Fill extrapolation edges with nearest neighbor
        if np.isnan(real_grid).any():
            real_near = scipy.interpolate.griddata(points, real_p, (grid_r, grid_c), method='nearest')
            imag_near = scipy.interpolate.griddata(points, imag_p, (grid_r, grid_c), method='nearest')
            mask = np.isnan(real_grid)
            real_grid[mask] = real_near[mask]
            imag_grid[mask] = imag_near[mask]

        H_li[n] = real_grid + 1j * imag_grid

    return H_li


def generate_pseudo_dataset(
    source_dir,
    target_dir,
    out_dir,
    beta_tau=None,
    delta_nu=None,
    snr_list=[-10, -5, 0, 5, 10, 15],
    num_samples=None,
    scale_mode="rms"
):
    """
    Generates pseudo dataset across all SNRs.
    """
    print("\n" + "=" * 80)
    print("  Approach A: Physical Pseudo-Dataset Generation (Delay Dilation & Doppler Modulation)")
    print("=" * 80)
    print(f"Source Directory : {source_dir}")
    print(f"Target Directory : {target_dir}")
    print(f"Output Directory : {out_dir}")
    print(f"Scaling Mode     : {scale_mode.upper()}")
    print(f"SNR Points (dB)  : {snr_list}")
    print(f"Num Samples      : {num_samples if num_samples is not None else 'All'}")
    print("=" * 80 + "\n")

    os.makedirs(out_dir, exist_ok=True)

    # 1. Discover SNR subfolders
    def find_snr_folder(root_dir, snr):
        candidates = [
            f"SNR_{snr}dB", f"SNR_{snr:+d}dB",
            f"{snr}dB", f"{snr:+d}dB",
            f"LS_{snr}dB", f"LS_{snr}"
        ]
        for c in candidates:
            p = os.path.join(root_dir, c)
            if os.path.isdir(p):
                return p
        # Check subdirectories matching integer snr
        for sub in os.listdir(root_dir):
            sub_p = os.path.join(root_dir, sub)
            if os.path.isdir(sub_p):
                # match digits
                import re
                m = re.search(r'[-+]?\d+', sub)
                if m and int(m.group()) == snr:
                    return sub_p
        return None

    # Pick a reference SNR to estimate beta_tau and delta_nu blindly if not provided
    ref_snr = 10 if 10 in snr_list else snr_list[0]
    src_ref_dir = find_snr_folder(source_dir, ref_snr) or source_dir
    tgt_ref_dir = find_snr_folder(target_dir, ref_snr) or target_dir

    print(f"[*] Loading reference datasets at SNR = {ref_snr} dB to estimate physical parameters...")
    H_src_ref, p_src_ref, r_p, c_p, _ = load_channel_mat(src_ref_dir, 'ls')
    H_tgt_ref, p_tgt_ref, _, _, _ = load_channel_mat(tgt_ref_dir, 'ls')

    if num_samples is not None:
        H_src_ref = H_src_ref[:num_samples]
        p_src_ref = p_src_ref[:num_samples]
        H_tgt_ref = H_tgt_ref[:num_samples]
        p_tgt_ref = p_tgt_ref[:num_samples]

    # Estimate or set beta_tau
    if beta_tau is None or beta_tau <= 0:
        b_est, s_info, t_info = estimate_delay_dilation_autocorr(p_src_ref, p_tgt_ref, r_p, c_p)
        beta_tau = b_est
        print(f"  [Auto-Estimate] Delay Dilation Beta_tau: {beta_tau:.3f} (Src lag: {s_info[2]:.1f}, Tgt lag: {t_info[2]:.1f})")
    else:
        print(f"  [Manual Override] Delay Dilation Beta_tau: {beta_tau:.3f}")

    # Estimate or set delta_nu
    if delta_nu is None:
        dnu_est, _, _ = estimate_bulk_doppler_shift(p_src_ref, p_tgt_ref, r_p, c_p)
        delta_nu = dnu_est
        print(f"  [Auto-Estimate] Relative Doppler Shift Delta_nu: {delta_nu:.2f} Hz")
    else:
        print(f"  [Manual Override] Relative Doppler Shift Delta_nu: {delta_nu:.2f} Hz")

    # Target RMS scaling calibration
    tgt_power = np.mean(np.abs(H_tgt_ref) ** 2)
    tgt_rms = np.sqrt(tgt_power)
    print(f"  [Calibration] Target Reference RMS: {tgt_rms:.6e}\n")

    summary_records = []

    # 2. Iterate across all requested SNRs
    for snr in snr_list:
        print(f"--- Processing SNR = {snr} dB ---")
        src_folder = find_snr_folder(source_dir, snr)
        tgt_folder = find_snr_folder(target_dir, snr)

        if src_folder is None:
            print(f"  [Warning] Source subfolder for SNR {snr} dB not found. Using reference source channels.")
            src_folder = src_ref_dir
        if tgt_folder is None:
            print(f"  [Warning] Target subfolder for SNR {snr} dB not found. Using reference target channels.")
            tgt_folder = tgt_ref_dir

        H_src, _, _, _, src_dict = load_channel_mat(src_folder, 'ls')
        _, _, _, _, tgt_dict = load_channel_mat(tgt_folder, 'ls')

        if num_samples is not None:
            H_src = H_src[:num_samples]

        N_s = H_src.shape[0]

        # Step A: Physical Mathematical Warping of Source Ground-Truth Labels
        t0 = time.perf_counter()
        H_pseudo_perf = warp_source_to_target(H_src, beta_tau, delta_nu)

        # Step B: Rescaling to match target domain energy level
        src_rms = np.sqrt(np.mean(np.abs(H_pseudo_perf) ** 2, axis=(1, 2), keepdims=True)) + 1e-30
        if scale_mode.lower() == "rms":
            H_pseudo_perf = H_pseudo_perf * (tgt_rms / src_rms)
        elif scale_mode.lower() == "standardize":
            tgt_mean = np.mean(H_tgt_ref)
            tgt_std = np.std(H_tgt_ref)
            src_mean = np.mean(H_pseudo_perf, axis=(1, 2), keepdims=True)
            src_std = np.std(H_pseudo_perf, axis=(1, 2), keepdims=True) + 1e-30
            H_pseudo_perf = (H_pseudo_perf - src_mean) / src_std * tgt_std + tgt_mean

        # Step C: 5G NR Transmission Simulation (Pilots + AWGN)
        # Extract true pilot values at pilot positions
        H_true_pilots = H_pseudo_perf[:, r_p, c_p]  # (N_s, 88)

        # Calibrate noise variance to pilot power
        pilot_power = np.mean(np.abs(H_true_pilots) ** 2)
        snr_linear = 10.0 ** (snr / 10.0)
        noise_var = pilot_power / snr_linear

        # Complex AWGN: N(0, noise_var/2) for real and imag
        noise = np.sqrt(noise_var / 2.0) * (
            np.random.randn(*H_true_pilots.shape) + 1j * np.random.randn(*H_true_pilots.shape)
        )
        H_ls_pilots = H_true_pilots + noise

        # Step D: Reconstruct baseline 2D Linear Interpolated Channel H_li
        H_li = interpolate_pilots_2d(H_ls_pilots, r_p, c_p, n_sc=132, n_sym=14)

        # Evaluate pseudo dataset baseline metrics
        nmse_li = compute_nmse(H_li, H_pseudo_perf)
        nmse_li_db = compute_nmse_db(H_li, H_pseudo_perf)
        ssim_li = compute_ssim_batch(H_li, H_pseudo_perf)

        elapsed = time.perf_counter() - t0
        print(f"  Generated {N_s} pseudo samples in {elapsed:.2f}s | LI NMSE: {nmse_li_db:.2f} dB, SSIM: {ssim_li:.4f}")

        # Step E: Save to nested folder structure matching repo conventions
        dest_snr_dir = os.path.join(out_dir, f"SNR_{snr}dB")
        os.makedirs(dest_snr_dir, exist_ok=True)
        dest_mat = os.path.join(dest_snr_dir, "matlabNTN.mat")

        # Save formatted dictionary: MATLAB expects (14, 132, N)
        save_vars = {
            'H_perfect': np.transpose(H_pseudo_perf, (2, 1, 0)),
            'H_li': np.transpose(H_li, (2, 1, 0)),
            'H_ls_pilots': H_ls_pilots,
            'pilot_rows': r_p + 1,  # 1-based MATLAB indices
            'pilot_cols': c_p + 1,  # 1-based MATLAB indices
            'pilot_indices': np.arange(1, len(r_p) + 1)[np.newaxis, :],
            'beta_tau': beta_tau,
            'delta_nu': delta_nu,
            'SNRdB': float(snr),
            'nmse_li': nmse_li,
            'nmse_li_db': nmse_li_db,
            'ssim_li': ssim_li
        }
        scipy.io.savemat(dest_mat, save_vars)
        print(f"  [Saved] -> {dest_mat}")

        summary_records.append({
            'snr': snr,
            'samples': N_s,
            'nmse_li_db': nmse_li_db,
            'ssim_li': ssim_li
        })

    # Save summary note and comparison plot
    plot_path = os.path.join(out_dir, "pseudo_dataset_comparison.png")
    plot_pseudo_samples(H_src[:4], H_pseudo_perf[:4], H_tgt_ref[:4], beta_tau, delta_nu, plot_path)

    note_path = os.path.join(out_dir, "note.md")
    with open(note_path, "w") as fh:
        fh.write("# Physical Pseudo-Dataset Generation Report\n\n")
        fh.write(f"- **Method:** Approach A (Analytical Delay Dilation & Doppler Modulation)\n")
        fh.write(f"- **Estimated Beta_tau:** `{beta_tau:.4f}`\n")
        fh.write(f"- **Estimated Delta_nu:** `{delta_nu:.2f} Hz`\n")
        fh.write(f"- **Scaling Mode:** `{scale_mode}`\n\n")
        fh.write("| SNR (dB) | Samples | 2D Linear Interp NMSE (dB) | SSIM |\n")
        fh.write("| :---: | :---: | :---: | :---: |\n")
        for rec in summary_records:
            fh.write(f"| {rec['snr']} | {rec['samples']} | {rec['nmse_li_db']:.2f} | {rec['ssim_li']:.4f} |\n")
        fh.write(f"\n![Comparison Plot](./pseudo_dataset_comparison.png)\n")

    print(f"\n[DONE] Pseudo dataset generated successfully at: {out_dir}\n")


def plot_pseudo_samples(H_src, H_pseudo, H_tgt, beta_tau, delta_nu, save_path):
    """
    Plots side-by-side comparison of Source, Warped Pseudo, and Target channels.
    """
    try:
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle(
            f"Physical Pseudo-Dataset Translation (Beta_tau = {beta_tau:.2f}, Delta_nu = {delta_nu:.1f} Hz)",
            fontsize=14, fontweight='bold'
        )

        for i in range(min(4, len(H_src))):
            # Col 0: Source Label
            ax0 = axes[i, 0]
            im0 = ax0.imshow(H_src[i].real, aspect='auto', cmap='coolwarm', origin='lower')
            ax0.set_title(f"Sample #{i} - Source (A100 100ns 70°)", fontsize=10)
            ax0.set_ylabel("Subcarriers", fontsize=9)
            fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

            # Col 1: Warped Pseudo Label
            ax1 = axes[i, 1]
            im1 = ax1.imshow(H_pseudo[i].real, aspect='auto', cmap='coolwarm', origin='lower')
            ax1.set_title(f"Sample #{i} - Pseudo (Dilated {beta_tau:.2f}x)", fontsize=10)
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # Col 2: Target Ground Truth (Reference)
            ax2 = axes[i, 2]
            im2 = ax2.imshow(H_tgt[i].real, aspect='auto', cmap='coolwarm', origin='lower')
            ax2.set_title(f"Sample #{i} - Target Reference (DUR300 300ns 30°)", fontsize=10)
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            if i == 3:
                ax0.set_xlabel("OFDM Symbols", fontsize=9)
                ax1.set_xlabel("OFDM Symbols", fontsize=9)
                ax2.set_xlabel("OFDM Symbols", fontsize=9)

        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  [Plot] Saved comparison figure -> {save_path}")
    except Exception as e:
        print(f"  [Warning] Failed to generate plot: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate physical pseudo dataset via Delay-Doppler coordinate warping.")
    parser.add_argument("--source-dir", type=str, default=r"generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz")
    parser.add_argument("--target-dir", type=str, default=r"generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps")
    parser.add_argument("--out-dir", type=str, default=r"math_translation\results\approachA_pseudo_dataset")
    parser.add_argument("--beta-tau", type=float, default=None, help="Delay dilation factor (auto-estimated if omitted)")
    parser.add_argument("--delta-nu", type=float, default=None, help="Relative Doppler shift in Hz (auto-estimated if omitted)")
    parser.add_argument("--snr-list", nargs="+", type=int, default=[-10, -5, 0, 5, 10, 15])
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--scale-mode", type=str, default="rms", choices=["rms", "standardize", "none"])

    args = parser.parse_args()
    generate_pseudo_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        out_dir=args.out_dir,
        beta_tau=args.beta_tau,
        delta_nu=args.delta_nu,
        snr_list=args.snr_list,
        num_samples=args.num_samples,
        scale_mode=args.scale_mode
    )

