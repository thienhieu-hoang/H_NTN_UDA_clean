"""
Dataset Review & Physical Characterization Tool
Analyzes a channel dataset folder (e.g. SNR_5dB, SNR_0dB) using the exact
native 132 x 14 Delay-Doppler (DD) grid (matching FDA / OTFS formulations),
and computes Delay Spread (tau_rms), Coherence Bandwidth (B_c), Doppler Shift,
and Coherence Time (T_c).
Outputs reports and figures in a subfolder report/.
"""

import os
import sys
import json
import argparse
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure math_translation is importable if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def load_dataset_folder(folder_path):
    """
    Robustly loads matlabNTN.mat or any candidate .mat from the folder.
    Supports MATLAB v7 and v7.3 HDF5.
    Returns:
        H_perfect: (N, 132, 14) complex array
        H_li: (N, 132, 14) or None
        H_pilots: (N, P) or None
        pilot_rows: (P,) or None
        pilot_cols: (P,) or None
        mat_file: path to loaded .mat
    """
    if not os.path.isdir(folder_path):
        if os.path.isfile(folder_path) and folder_path.endswith('.mat'):
            mat_file = folder_path
            folder_path = os.path.dirname(folder_path)
        else:
            raise FileNotFoundError(f"Directory not found: {folder_path}")
    else:
        candidates = ["matlabNTN.mat", "channel_dur_randomizedUE.mat"]
        mat_file = None
        for c in candidates:
            p = os.path.join(folder_path, c)
            if os.path.isfile(p):
                mat_file = p
                break
        if mat_file is None:
            mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat') and not f.startswith('inferredChannel')]
            if mat_files:
                mat_file = os.path.join(folder_path, mat_files[0])
            else:
                raise FileNotFoundError(f"No MAT file found in directory: {folder_path}")

    try:
        mat = scipy.io.loadmat(mat_file)
        is_hdf5 = False
    except NotImplementedError:
        import h5py
        is_hdf5 = True

    H_li = None
    H_pilots = None
    pilot_rows = None
    pilot_cols = None

    if is_hdf5:
        import h5py
        with h5py.File(mat_file, 'r') as f:
            def h5_comp(ds):
                val = ds[()]
                if isinstance(val, np.ndarray) and val.dtype.names is not None:
                    if 'real' in val.dtype.names and 'imag' in val.dtype.names:
                        return val['real'] + 1j * val['imag']
                return val

            H_perfect = h5_comp(f['H_perfect'])
            if 'H_li' in f:
                H_li = h5_comp(f['H_li'])
            if 'H_ls_pilots' in f:
                H_pilots = h5_comp(f['H_ls_pilots'])
            if 'pilot_rows' in f:
                pilot_rows = np.asarray(f['pilot_rows'][()]).squeeze()
            if 'pilot_cols' in f:
                pilot_cols = np.asarray(f['pilot_cols'][()]).squeeze()
    else:
        H_perfect = mat['H_perfect']
        if 'H_li' in mat:
            H_li = mat['H_li']
        if 'H_ls_pilots' in mat:
            H_pilots = mat['H_ls_pilots']
        if 'pilot_rows' in mat:
            pilot_rows = np.asarray(mat['pilot_rows']).squeeze()
        if 'pilot_cols' in mat:
            pilot_cols = np.asarray(mat['pilot_cols']).squeeze()

    # Transpose to standard (N, 132, 14)
    if H_perfect.ndim == 3:
        if H_perfect.shape[0] == 14 and H_perfect.shape[1] == 132:
            H_perfect = np.transpose(H_perfect, (2, 1, 0))
        elif H_perfect.shape[0] == 132 and H_perfect.shape[1] == 14:
            H_perfect = np.transpose(H_perfect, (2, 0, 1))

    if H_li is not None and H_li.ndim == 3:
        if H_li.shape[0] == 14 and H_li.shape[1] == 132:
            H_li = np.transpose(H_li, (2, 1, 0))
        elif H_li.shape[0] == 132 and H_li.shape[1] == 14:
            H_li = np.transpose(H_li, (2, 0, 1))

    return H_perfect, H_li, H_pilots, pilot_rows, pilot_cols, mat_file


def compute_dataset_physics(H, H_li=None, scs_khz=30.0):
    """
    Computes exact native 132x14 Delay-Doppler representation matching FDA / OTFS,
    along with Power Delay Profile (PDP), Doppler Spectrum, and ACFs.
    H shape: (N, 132, 14)
    """
    N, n_sc, n_sym = H.shape
    scs_hz = scs_khz * 1e3
    
    # 5G NR slot with 30 kHz SCS: 14 symbols in 0.5 ms -> Tsym = 0.5 ms / 14
    t_slot_s = 0.5e-3 if scs_khz == 30.0 else 1.0e-3
    t_sym_s = t_slot_s / float(n_sym)

    # --------------------------------------------------------------------------
    # 1. Native 132x14 Delay-Doppler Transformation (Matching F_extract_DD)
    # --------------------------------------------------------------------------
    # Step 1: IFFT along Subcarriers (axis 1: Frequency -> Delay, exact 132 points)
    grid_delay = np.fft.ifft(H, axis=1)
    
    # Step 2: FFT along Symbols (axis 2: Time -> Doppler, exact 14 points)
    grid_dd_raw = np.fft.fft(grid_delay, axis=2)
    
    # Step 3: Shift both dimensions to center (Delay=0 at center, Doppler=0 at center)
    H_dd = np.fft.fftshift(grid_dd_raw, axes=(1, 2))  # (N, 132, 14)

    # 2D Average Delay-Doppler Power Matrix (132 x 14)
    P_dd = np.mean(np.abs(H_dd) ** 2, axis=0)  # (132, 14)

    # Native OTFS / DD Bins
    # Delay bins: -66 to +65 (center is at 0)
    delay_bins = np.arange(-n_sc // 2, n_sc // 2)
    # Delay resolution: dt = 1 / (N_sc * SCS) = 1 / (132 * 30 kHz) ≈ 252.525 ns
    tau_res_ns = (1.0 / (float(n_sc) * scs_hz)) * 1e9
    tau_axis_ns = delay_bins * tau_res_ns

    # Doppler bins: -7 to +6 (center is at 0)
    doppler_bins = np.arange(-n_sym // 2, n_sym // 2)
    # Doppler resolution: dnu = 1 / (N_sym * Tsym) = 1 / (0.5 ms) = 2000 Hz
    dnu_res_hz = 1.0 / (float(n_sym) * t_sym_s)
    nu_axis_hz = doppler_bins * dnu_res_hz

    # --------------------------------------------------------------------------
    # 2. Power Delay Profile (PDP) in Causal Delay Domain [0, tau_max]
    # --------------------------------------------------------------------------
    # In causal time domain h(tau, t) = ifft(H, axis=1), delay index 0 is first arrival
    h_causal = grid_delay  # (N, 132, 14)
    pdp_raw = np.mean(np.abs(h_causal) ** 2, axis=(0, 2))  # (132,)

    # Restrict to meaningful multipath channel region (first 25 subcarrier delay taps)
    max_taps = min(25, n_sc)
    pdp = pdp_raw[:max_taps]
    tau_causal_ns = np.arange(max_taps) * tau_res_ns

    # Compute tau_mean and tau_rms
    pdp_sum = np.sum(pdp) + 1e-30
    tau_mean_ns = np.sum(tau_causal_ns * pdp) / pdp_sum
    tau_rms_ns = np.sqrt(np.sum(((tau_causal_ns - tau_mean_ns) ** 2) * pdp) / pdp_sum)

    pdp_db = 10.0 * np.log10(pdp / (np.max(pdp) + 1e-30) + 1e-30)

    # --------------------------------------------------------------------------
    # 3. Doppler Power Spectrum (14 bins centered at 0)
    # --------------------------------------------------------------------------
    doppler_spectrum = np.sum(P_dd, axis=0)  # (14,)
    dop_sum = np.sum(doppler_spectrum) + 1e-30
    nu_mean_hz = np.sum(nu_axis_hz * doppler_spectrum) / dop_sum
    nu_rms_hz = np.sqrt(np.sum(((nu_axis_hz - nu_mean_hz) ** 2) * doppler_spectrum) / dop_sum)
    dop_db = 10.0 * np.log10(doppler_spectrum / (np.max(doppler_spectrum) + 1e-30) + 1e-30)

    # --------------------------------------------------------------------------
    # 4. Frequency Autocorrelation Function R_f(Delta_k)
    # --------------------------------------------------------------------------
    max_k_lag = min(50, n_sc - 1)
    rf_lags = np.arange(max_k_lag + 1)
    rf_vals = np.zeros(max_k_lag + 1, dtype=float)

    for l in range(max_k_lag + 1):
        prod = H[:, l:, :] * np.conj(H[:, :n_sc - l, :])
        rf_vals[l] = np.abs(np.mean(prod))

    if rf_vals[0] > 1e-30:
        rf_vals = rf_vals / rf_vals[0]

    def find_crossing_lag(lags, vals, thresh):
        for i in range(len(vals) - 1):
            if vals[i] >= thresh and vals[i + 1] < thresh:
                slope = (vals[i + 1] - vals[i])
                return lags[i] + (thresh - vals[i]) / slope if abs(slope) > 1e-12 else float(lags[i])
        return float(lags[-1]) if vals[-1] >= thresh else float(lags[0])

    bc_lag_05 = find_crossing_lag(rf_lags, rf_vals, 0.5)
    bc_khz_05 = bc_lag_05 * scs_khz
    bc_lag_09 = find_crossing_lag(rf_lags, rf_vals, 0.9)
    bc_khz_09 = bc_lag_09 * scs_khz

    # --------------------------------------------------------------------------
    # 5. Time Autocorrelation Function R_t(Delta_l)
    # --------------------------------------------------------------------------
    max_t_lag = n_sym - 1
    rt_lags = np.arange(max_t_lag + 1)
    rt_vals = np.zeros(max_t_lag + 1, dtype=float)

    for l in range(max_t_lag + 1):
        prod = H[:, :, l:] * np.conj(H[:, :, :n_sym - l])
        rt_vals[l] = np.abs(np.mean(prod))

    if rt_vals[0] > 1e-30:
        rt_vals = rt_vals / rt_vals[0]

    tc_lag_05 = find_crossing_lag(rt_lags, rt_vals, 0.5)
    tc_ms_05 = tc_lag_05 * t_sym_s * 1e3

    # --------------------------------------------------------------------------
    # 6. Channel Power & Dynamic Range
    # --------------------------------------------------------------------------
    mag_sq = np.abs(H) ** 2
    avg_power = float(np.mean(mag_sq))
    rms_power = float(np.sqrt(avg_power))
    min_power = float(np.min(mag_sq) + 1e-30)
    max_power = float(np.max(mag_sq))
    dynamic_range_db = float(10.0 * np.log10(max_power / min_power))

    # --------------------------------------------------------------------------
    # 7. Baseline LI Interpolation Metrics (if present)
    # --------------------------------------------------------------------------
    nmse_li_db = None
    ssim_li = None
    if H_li is not None and H_li.shape == H.shape:
        err = np.sum(np.abs(H_li - H) ** 2)
        ref = np.sum(np.abs(H) ** 2) + 1e-30
        nmse_li = float(err / ref)
        nmse_li_db = float(10.0 * np.log10(nmse_li))

        from scipy.ndimage import uniform_filter
        def ssim_slice(img1, img2):
            ux = uniform_filter(img1, size=7)
            uy = uniform_filter(img2, size=7)
            uxx = uniform_filter(img1 * img1, size=7)
            uyy = uniform_filter(img2 * img2, size=7)
            uxy = uniform_filter(img1 * img2, size=7)
            vx = uxx - ux * ux
            vy = uyy - uy * uy
            vxy = uxy - ux * uy
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            num = (2 * ux * uy + c1) * (2 * vxy + c2)
            den = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
            return float(np.mean(num / (den + 1e-30)))

        ssim_vals = [ssim_slice(np.abs(H_li[i]), np.abs(H[i])) for i in range(min(50, N))]
        ssim_li = float(np.mean(ssim_vals))

    return {
        'N': N,
        'n_sc': n_sc,
        'n_sym': n_sym,
        'scs_khz': scs_khz,
        'tau_rms_ns': float(tau_rms_ns),
        'tau_mean_ns': float(tau_mean_ns),
        'tau_res_ns': float(tau_res_ns),
        'dnu_res_hz': float(dnu_res_hz),
        'bc_lag_05': float(bc_lag_05),
        'bc_khz_05': float(bc_khz_05),
        'bc_lag_09': float(bc_lag_09),
        'bc_khz_09': float(bc_khz_09),
        'nu_mean_hz': float(nu_mean_hz),
        'nu_rms_hz': float(nu_rms_hz),
        'tc_lag_05': float(tc_lag_05),
        'tc_ms_05': float(tc_ms_05),
        'avg_power': avg_power,
        'rms_power': rms_power,
        'dynamic_range_db': dynamic_range_db,
        'nmse_li_db': nmse_li_db,
        'ssim_li': ssim_li,
        # Exact 132 x 14 DD arrays
        'P_dd': P_dd,
        'delay_bins': delay_bins,
        'tau_axis_ns': tau_axis_ns,
        'doppler_bins': doppler_bins,
        'nu_axis_hz': nu_axis_hz,
        # 1D arrays
        'tau_causal_ns': tau_causal_ns,
        'pdp_db': pdp_db,
        'dop_db': dop_db,
        'rf_lags': rf_lags,
        'rf_vals': rf_vals,
        'rt_lags': rt_lags,
        'rt_vals': rt_vals
    }


def generate_plots_and_report(H_sample, metrics, folder_path, report_dir):
    """
    Renders 2D DD heatmap (native 132x14 centered grid), 1D curves, and report.md.
    """
    os.makedirs(report_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # Figure 1: 2D Delay-Doppler Energy Heatmap (Native 132 x 14 Grid)
    # --------------------------------------------------------------------------
    fig1, (ax1, ax1_zoom) = plt.subplots(1, 2, figsize=(14, 6))

    P_dd = metrics['P_dd']  # (132, 14)
    P_dd_db = 10.0 * np.log10(P_dd / (np.max(P_dd) + 1e-30) + 1e-30)
    P_dd_db = np.clip(P_dd_db, -35.0, 0.0)

    d_bins = metrics['delay_bins']      # -66 to 65
    dop_bins = metrics['doppler_bins']  # -7 to 6

    # Full 132 x 14 DD Plane
    im1 = ax1.imshow(
        P_dd_db,
        aspect='auto',
        origin='lower',
        extent=[dop_bins[0], dop_bins[-1], d_bins[0], d_bins[-1]],
        cmap='inferno'
    )
    fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Normalized Power (dB)")
    N_total = metrics['N']
    ax1.set_title(f"2D Delay-Doppler Grid (132 × 14 Bins)\n[Ensemble Average over all N = {N_total} samples]", fontsize=11, fontweight='bold')
    ax1.set_xlabel(f"Doppler Bins (Δν = {metrics['dnu_res_hz']:.0f} Hz / bin)", fontsize=10)
    ax1.set_ylabel(f"Delay Bins (Δτ = {metrics['tau_res_ns']:.1f} ns / bin)", fontsize=10)
    ax1.axhline(0, color='cyan', linestyle=':', alpha=0.6)
    ax1.axvline(0, color='cyan', linestyle=':', alpha=0.6)
    ax1.grid(True, linestyle=':', alpha=0.3, color='white')

    # Zoomed Center Region (Matching FDA 13 x 5 Window Area)
    # Center rows: 66 ± 12, Center cols: 7 ± 4
    cy, cx = len(d_bins) // 2, len(dop_bins) // 2
    zoom_y = slice(max(0, cy - 12), min(len(d_bins), cy + 13))
    zoom_x = slice(max(0, cx - 4), min(len(dop_bins), cx + 5))

    im2 = ax1_zoom.imshow(
        P_dd_db[zoom_y, zoom_x],
        aspect='auto',
        origin='lower',
        extent=[dop_bins[zoom_x.start], dop_bins[zoom_x.stop - 1], d_bins[zoom_y.start], d_bins[zoom_y.stop - 1]],
        cmap='inferno'
    )
    fig1.colorbar(im2, ax=ax1_zoom, fraction=0.046, pad=0.04, label="Normalized Power (dB)")
    ax1_zoom.set_title("Center Low-Frequency Delay-Doppler Core", fontsize=11, fontweight='bold')
    ax1_zoom.set_xlabel(f"Doppler Bins (Center = 0 Hz)", fontsize=10)
    ax1_zoom.set_ylabel(f"Delay Bins (Center = 0 ns)", fontsize=10)
    ax1_zoom.axhline(0, color='cyan', linestyle=':', alpha=0.8)
    ax1_zoom.axvline(0, color='cyan', linestyle=':', alpha=0.8)
    ax1_zoom.grid(True, linestyle=':', alpha=0.4, color='white')

    fig1_path = os.path.join(report_dir, "fig_delay_doppler_2d.png")
    fig1.tight_layout()
    fig1.savefig(fig1_path, dpi=200, bbox_inches='tight')
    plt.close(fig1)

    # --------------------------------------------------------------------------
    # Figure 2: 4-Panel 1D Physical Characterization Curves
    # --------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(f"1D Physical Channel Characterization (Ensemble Average over all N = {metrics['N']} samples)", fontsize=13, fontweight='bold')

    # Panel (a): Power Delay Profile (PDP)
    ax_a = axes2[0, 0]
    ax_a.plot(metrics['tau_causal_ns'], metrics['pdp_db'], marker='o', markersize=4, color='#cf222e', lw=2, label="PDP (dB)")
    ax_a.axvline(metrics['tau_rms_ns'], color='blue', linestyle='--', label=f"τ_rms = {metrics['tau_rms_ns']:.1f} ns")
    ax_a.set_title("Power Delay Profile P(τ)", fontsize=11, fontweight='bold')
    ax_a.set_xlabel("Propagation Delay τ (ns)", fontsize=10)
    ax_a.set_ylabel("Power (dB)", fontsize=10)
    ax_a.set_ylim(-35, 2)
    ax_a.grid(True, linestyle='--', alpha=0.5)
    ax_a.legend(loc='upper right', fontsize=9)

    # Panel (b): Doppler Power Spectrum S(nu)
    ax_b = axes2[0, 1]
    ax_b.plot(metrics['nu_axis_hz'], metrics['dop_db'], marker='s', markersize=4, color='#0969da', lw=2, label="Doppler Spectrum (dB)")
    ax_b.axvline(metrics['nu_mean_hz'], color='red', linestyle='--', label=f"ν_mean = {metrics['nu_mean_hz']:.1f} Hz")
    ax_b.set_title("Doppler Power Spectrum S(ν) [14 Bins]", fontsize=11, fontweight='bold')
    ax_b.set_xlabel("Doppler Frequency ν (Hz)", fontsize=10)
    ax_b.set_ylabel("Power (dB)", fontsize=10)
    ax_b.set_ylim(-35, 2)
    ax_b.grid(True, linestyle='--', alpha=0.5)
    ax_b.legend(loc='upper right', fontsize=9)

    # Panel (c): Frequency Autocorrelation Function |Rf(Delta_k)|
    ax_c = axes2[1, 0]
    ax_c.plot(metrics['rf_lags'], metrics['rf_vals'], color='#1a7f37', lw=2, label="|R_f(Δk)|")
    ax_c.axhline(0.5, color='gray', linestyle='--', label="Threshold = 0.5")
    ax_c.axvline(metrics['bc_lag_05'], color='orange', linestyle=':', label=f"B_c = {metrics['bc_khz_05']:.0f} kHz ({metrics['bc_lag_05']:.1f} sc)")
    ax_c.set_title("Frequency Autocorrelation (Coherence BW)", fontsize=11, fontweight='bold')
    ax_c.set_xlabel("Subcarrier Lag Δk (30 kHz per lag)", fontsize=10)
    ax_c.set_ylabel("Normalized Correlation", fontsize=10)
    ax_c.set_ylim(0, 1.05)
    ax_c.grid(True, linestyle='--', alpha=0.5)
    ax_c.legend(loc='upper right', fontsize=9)

    # Panel (d): Time Autocorrelation Function |Rt(Delta_l)|
    ax_d = axes2[1, 1]
    ax_d.plot(metrics['rt_lags'], metrics['rt_vals'], color='#8250df', lw=2, label="|R_t(Δl)|")
    ax_d.axhline(0.5, color='gray', linestyle='--', label="Threshold = 0.5")
    ax_d.axvline(metrics['tc_lag_05'], color='red', linestyle=':', label=f"T_c = {metrics['tc_ms_05']:.2f} ms ({metrics['tc_lag_05']:.1f} sym)")
    ax_d.set_title("Time Autocorrelation (Coherence Time)", fontsize=11, fontweight='bold')
    ax_d.set_xlabel("OFDM Symbol Lag Δl", fontsize=10)
    ax_d.set_ylabel("Normalized Correlation", fontsize=10)
    ax_d.set_ylim(0, 1.05)
    ax_d.grid(True, linestyle='--', alpha=0.5)
    ax_d.legend(loc='upper right', fontsize=9)

    fig2_path = os.path.join(report_dir, "fig_channel_physics_1d.png")
    fig2.tight_layout()
    fig2.savefig(fig2_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)

    # --------------------------------------------------------------------------
    # Figure 3: Sample #0 Time-Frequency Heatmaps
    # --------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
    h0 = H_sample[0]

    im_r = axes3[0].imshow(h0.real, aspect='auto', cmap='coolwarm', origin='lower')
    axes3[0].set_title("Sample #0 - Real Part", fontsize=10)
    axes3[0].set_xlabel("OFDM Symbols", fontsize=9)
    axes3[0].set_ylabel("Subcarriers", fontsize=9)
    fig3.colorbar(im_r, ax=axes3[0], fraction=0.046, pad=0.04)

    im_i = axes3[1].imshow(h0.imag, aspect='auto', cmap='coolwarm', origin='lower')
    axes3[1].set_title("Sample #0 - Imaginary Part", fontsize=10)
    axes3[1].set_xlabel("OFDM Symbols", fontsize=9)
    fig3.colorbar(im_i, ax=axes3[1], fraction=0.046, pad=0.04)

    im_m = axes3[2].imshow(np.abs(h0), aspect='auto', cmap='viridis', origin='lower')
    axes3[2].set_title("Sample #0 - Absolute Magnitude", fontsize=10)
    axes3[2].set_xlabel("OFDM Symbols", fontsize=9)
    fig3.colorbar(im_m, ax=axes3[2], fraction=0.046, pad=0.04)

    fig3_path = os.path.join(report_dir, "fig_samples_tf_heatmaps.png")
    fig3.tight_layout()
    fig3.savefig(fig3_path, dpi=200, bbox_inches='tight')
    plt.close(fig3)

    # --------------------------------------------------------------------------
    # Save machine-readable JSON metrics
    # --------------------------------------------------------------------------
    json_metrics = {
        'folder_path': os.path.abspath(folder_path),
        'N': metrics['N'],
        'n_sc': metrics['n_sc'],
        'n_sym': metrics['n_sym'],
        'scs_khz': metrics['scs_khz'],
        'tau_rms_ns': metrics['tau_rms_ns'],
        'tau_mean_ns': metrics['tau_mean_ns'],
        'tau_res_ns': metrics['tau_res_ns'],
        'dnu_res_hz': metrics['dnu_res_hz'],
        'bc_khz_05': metrics['bc_khz_05'],
        'bc_lag_05': metrics['bc_lag_05'],
        'nu_mean_hz': metrics['nu_mean_hz'],
        'nu_rms_hz': metrics['nu_rms_hz'],
        'tc_ms_05': metrics['tc_ms_05'],
        'tc_lag_05': metrics['tc_lag_05'],
        'avg_power': metrics['avg_power'],
        'rms_power': metrics['rms_power'],
        'dynamic_range_db': metrics['dynamic_range_db'],
        'nmse_li_db': metrics['nmse_li_db'],
        'ssim_li': metrics['ssim_li']
    }
    json_path = os.path.join(report_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_metrics, f, indent=4)

    # --------------------------------------------------------------------------
    # Generate Markdown Report (report.md)
    # --------------------------------------------------------------------------
    folder_name = os.path.basename(os.path.normpath(folder_path))
    parent_name = os.path.basename(os.path.dirname(os.path.normpath(folder_path)))

    report_md_path = os.path.join(report_dir, "report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Channel Dataset Physical Characterization Report\n\n")
        f.write(f"- **Target Folder:** `{parent_name}/{folder_name}`\n")
        f.write(f"- **Absolute Path:** `{os.path.abspath(folder_path)}`\n")
        f.write(f"- **Total Realizations:** `{metrics['N']}`\n")
        f.write(f"- **Grid Size:** `{metrics['n_sc']} Subcarriers × {metrics['n_sym']} Symbols` (SCS = `{metrics['scs_khz']} kHz`)\n")
        f.write(f"- **Delay-Doppler Resolution:** `Δτ = {metrics['tau_res_ns']:.2f} ns/bin`, `Δν = {metrics['dnu_res_hz']:.1f} Hz/bin`\n\n")

        f.write("## 1. Physical Propagation & Coherence Metrics\n\n")
        f.write("| Physical Parameter | Value | Standard Interpretation |\n")
        f.write("| :--- | :---: | :--- |\n")
        f.write(f"| **RMS Delay Spread ($\\tau_{{\\text{{rms}}}}$)** | `{metrics['tau_rms_ns']:.2f} ns` | Multipath dispersion along subcarriers |\n")
        f.write(f"| **Mean Excess Delay ($\\bar{{\\tau}}$)** | `{metrics['tau_mean_ns']:.2f} ns` | Center of multipath power distribution |\n")
        f.write(f"| **Coherence Bandwidth ($B_{{c, 0.5}}$)** | `{metrics['bc_khz_05']:.1f} kHz` (`{metrics['bc_lag_05']:.2f}` sc) | Frequency lag where correlation drops to 0.5 |\n")
        f.write(f"| **Mean Doppler Shift ($\\bar{{\\nu}}$)** | `{metrics['nu_mean_hz']:.2f} Hz` | Bulk orbital/velocity Doppler frequency |\n")
        f.write(f"| **RMS Doppler Spread ($\\nu_{{\\text{{rms}}}}$)** | `{metrics['nu_rms_hz']:.2f} Hz` | Temporal fading velocity across OFDM symbols |\n")
        f.write(f"| **Coherence Time ($T_{{c, 0.5}}$)** | `{metrics['tc_ms_05']:.2f} ms` (`{metrics['tc_lag_05']:.2f}` sym) | Time lag where correlation drops to 0.5 |\n")
        f.write(f"| **Channel Power (RMS)** | `{metrics['rms_power']:.4e}` | Average energy magnitude |\n")
        f.write(f"| **Dynamic Range** | `{metrics['dynamic_range_db']:.2f} dB` | Peak-to-minimum power variation |\n")

        if metrics['nmse_li_db'] is not None:
            f.write(f"\n## 2. Baseline Pilot Estimation Performance\n\n")
            f.write("| Baseline Estimator | NMSE (dB) | SSIM |\n")
            f.write("| :--- | :---: | :---: |\n")
            ssim_str = f"{metrics['ssim_li']:.4f}" if metrics['ssim_li'] is not None else "N/A"
            f.write(f"| **2D Linear Interpolation ($H_{{\\text{{li}}}}$)** | `{metrics['nmse_li_db']:.2f} dB` | `{ssim_str}` |\n")

        f.write("\n## 3. Visual Characterization Plots\n\n")
        f.write(f"### 2D Delay-Doppler Energy Distribution (Ensemble Average over all N = {metrics['N']} samples)\n")
        f.write("![2D Delay-Doppler](./fig_delay_doppler_2d.png)\n\n")
        f.write(f"### 1D Physical Curves (Ensemble Average over all N = {metrics['N']} samples)\n")
        f.write("![1D Physics](./fig_channel_physics_1d.png)\n\n")
        f.write("### Single Realization Snapshot (Sample #0 in Time-Frequency Domain)\n")
        f.write("![TF Heatmaps](./fig_samples_tf_heatmaps.png)\n")

    print(f"\n[SUCCESS] Report successfully generated in: {report_dir}")
    print(f"  - Markdown Report : {report_md_path}")
    print(f"  - JSON Metrics    : {json_path}")
    print(f"  - 2D DD Figure    : {fig1_path}")
    print(f"  - 1D Physics Fig  : {fig2_path}")


def review_dataset(dataset_dir, out_dir=None, scs_khz=30.0, num_samples=None):
    """
    Main review function for a given dataset directory.
    """
    print("=" * 80)
    print("  Channel Dataset Physical Characterization & Review Tool")
    print("=" * 80)
    print(f"Target Directory : {dataset_dir}")

    H_perfect, H_li, H_pilots, pilot_rows, pilot_cols, mat_file = load_dataset_folder(dataset_dir)
    print(f"Loaded MAT File  : {mat_file}")
    print(f"Loaded Shape     : {H_perfect.shape} (Samples, Subcarriers, Symbols)")

    if num_samples is not None and num_samples < H_perfect.shape[0]:
        H_perfect = H_perfect[:num_samples]
        if H_li is not None:
            H_li = H_li[:num_samples]
        print(f"Restricted to    : {num_samples} samples")

    if out_dir is None:
        out_dir = os.path.join(dataset_dir, "report")

    print(f"Output Directory : {out_dir}")
    print("Computing native 132x14 Delay-Doppler transformations and physical metrics...")

    metrics = compute_dataset_physics(H_perfect, H_li=H_li, scs_khz=scs_khz)
    print(f"  [Calculated] RMS Delay Spread  : {metrics['tau_rms_ns']:.2f} ns")
    print(f"  [Calculated] Coherence BW (0.5): {metrics['bc_khz_05']:.1f} kHz ({metrics['bc_lag_05']:.2f} subcarriers)")
    print(f"  [Calculated] Mean Doppler Shift: {metrics['nu_mean_hz']:.2f} Hz")
    print(f"  [Calculated] RMS Doppler Spread: {metrics['nu_rms_hz']:.2f} Hz")
    print(f"  [Calculated] Coherence Time    : {metrics['tc_ms_05']:.2f} ms")

    generate_plots_and_report(H_perfect, metrics, dataset_dir, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate physical characterization report for a channel dataset folder.")
    parser.add_argument("--dataset-dir", type=str, default=r"generatedChan_mathPseudo\A100_DUR300_LS\SNR_5dB",
                        help="Path to dataset directory containing matlabNTN.mat (e.g. SNR_5dB, SNR_0dB)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output report directory (defaults to <dataset-dir>/report)")
    parser.add_argument("--scs-khz", type=float, default=30.0,
                        help="Subcarrier spacing in kHz (default: 30.0)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")

    args = parser.parse_args()
    review_dataset(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        scs_khz=args.scs_khz,
        num_samples=args.num_samples
    )
