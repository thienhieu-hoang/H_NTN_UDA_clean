"""
========================================================================================
FDA Window and Delay-Doppler Blending Utilities for Channel Estimation
========================================================================================
Implements:
1. 2D FDA mixing masks:
   - 'hard': Legacy binary rectangular boxcar window (13 x w_w).
   - 'tukey_outward': Smooth outward raised-cosine tapering window (100% target core,
                      outward cosine taper, eliminating Gibbs ringing artifacts).
   - 'tukey_inward': Smooth inward raised-cosine tapering window.
   - 'supergaussian': Smooth flat-top elliptical super-Gaussian window.
2. 2D Delay-Doppler transform and Fourier amplitude blending.
3. Precomputation of source domain reference amplitude spectrum (spectral prototype).
4. Pilot Consistency Projection (PCP) for OFDM channel grids.
========================================================================================
"""

import os
import re
import numpy as np
import scipy.io as sio
import h5py


def create_fda_window(
    grid_h: int = 132,
    grid_w: int = 14,
    win_h_px: int = 13,
    win_w_px: int = 3,
    window_type: str = "tukey_outward",
    taper_h: int = 2,
    taper_w: int = 1,
) -> np.ndarray:
    """
    Generates a 2D Delay-Doppler FDA mixing mask M in [0, 1].

    Parameters
    ----------
    grid_h : int
        Total subcarriers / delay bins (default 132).
    grid_w : int
        Total OFDM symbols / Doppler bins (default 14).
    win_h_px : int
        Core window height along Delay (default 13).
    win_w_px : int
        Core window width along Doppler (default 3 or 5).
    window_type : str
        'hard': Legacy binary rectangular boxcar window.
        'tukey_outward': 100% target core, raised-cosine outward taper (Recommended).
        'tukey_inward': Raised-cosine inward taper inside core.
        'supergaussian': Elliptical flat-top super-Gaussian.
    taper_h : int
        Number of transition rows along Delay (default 2).
    taper_w : int
        Number of transition columns along Doppler (default 1).

    Returns
    -------
    mask : np.ndarray of shape (grid_h, grid_w), dtype np.float32
    """
    cy = grid_h // 2  # 66 for 132
    cx = grid_w // 2  # 7 for 14

    r_h = win_h_px // 2  # 6 for 13
    r_w = win_w_px // 2  # 1 for 3, 2 for 5

    window_type = str(window_type).lower().strip()

    if window_type in ("tukey_outward", "tukey", "smooth", "raised_cosine"):
        # 1D Delay profile (Subcarriers)
        y_idx = np.arange(grid_h)
        d_y = np.abs(y_idx - cy)
        w_y = np.zeros(grid_h, dtype=np.float32)

        core_mask_y = d_y <= r_h
        taper_mask_y = (d_y > r_h) & (d_y <= (r_h + taper_h))
        k_y = d_y[taper_mask_y] - r_h

        w_y[core_mask_y] = 1.0
        w_y[taper_mask_y] = 0.5 * (1.0 + np.cos(np.pi * k_y / (taper_h + 1)))

        # 1D Doppler profile (OFDM Symbols)
        x_idx = np.arange(grid_w)
        d_x = np.abs(x_idx - cx)
        w_x = np.zeros(grid_w, dtype=np.float32)

        core_mask_x = d_x <= r_w
        taper_mask_x = (d_x > r_w) & (d_x <= (r_w + taper_w))
        k_x = d_x[taper_mask_x] - r_w

        w_x[core_mask_x] = 1.0
        w_x[taper_mask_x] = 0.5 * (1.0 + np.cos(np.pi * k_x / (taper_w + 1)))

        # 2D Separable Outer Product: (H, 1) x (1, W) -> (H, W)
        mask = w_y[:, None] * w_x[None, :]

    elif window_type == "tukey_inward":
        # Taper occurs inside the nominal win_h x win_w box
        y_idx = np.arange(grid_h)
        d_y = np.abs(y_idx - cy)
        w_y = np.zeros(grid_h, dtype=np.float32)
        r_inner_y = max(0, r_h - taper_h)
        w_y[d_y <= r_inner_y] = 1.0
        in_taper_y = (d_y > r_inner_y) & (d_y <= r_h)
        k_y = d_y[in_taper_y] - r_inner_y
        w_y[in_taper_y] = 0.5 * (1.0 + np.cos(np.pi * k_y / (taper_h + 1)))

        x_idx = np.arange(grid_w)
        d_x = np.abs(x_idx - cx)
        w_x = np.zeros(grid_w, dtype=np.float32)
        r_inner_x = max(0, r_w - taper_w)
        w_x[d_x <= r_inner_x] = 1.0
        in_taper_x = (d_x > r_inner_x) & (d_x <= r_w)
        k_x = d_x[in_taper_x] - r_inner_x
        w_x[in_taper_x] = 0.5 * (1.0 + np.cos(np.pi * k_x / (taper_w + 1)))

        mask = w_y[:, None] * w_x[None, :]

    elif window_type == "supergaussian":
        y_idx, x_idx = np.ogrid[:grid_h, :grid_w]
        norm_dist = ((y_idx - cy) / max(1, r_h)) ** 2 + ((x_idx - cx) / max(1, r_w)) ** 2
        mask = np.exp(-(norm_dist ** 2)).astype(np.float32)
        mask[mask < 0.01] = 0.0

    elif window_type in ("hard", "boxcar", "rect", "rectangular"):
        # Legacy rectangular boxcar window
        mask = np.zeros((grid_h, grid_w), dtype=np.float32)
        y_start = max(0, cy - r_h)
        y_end = min(grid_h, cy + r_h + 1)
        x_start = max(0, cx - r_w)
        x_end = min(grid_w, cx + r_w + 1)
        mask[y_start:y_end, x_start:x_end] = 1.0

    else:
        raise ValueError(f"Unsupported window_type '{window_type}'. Choose from 'hard', 'tukey_outward', 'tukey_inward', 'supergaussian'.")

    return mask.astype(np.float32)


def tf_to_delay_doppler(H_tf: np.ndarray) -> np.ndarray:
    """
    Transforms Time-Frequency channel grid to Delay-Doppler domain.
    Matches MATLAB:
      1. Frequency -> Delay via IFFT along axis -2 (subcarriers)
      2. Time -> Doppler via FFT along axis -1 (symbols)
      3. fftshift on both axes
    """
    grid_delay = np.fft.ifft(H_tf, axis=-2)
    grid_dd_raw = np.fft.fft(grid_delay, axis=-1)
    grid_dd = np.fft.fftshift(grid_dd_raw, axes=(-2, -1))
    return grid_dd


def delay_doppler_to_tf(H_dd: np.ndarray) -> np.ndarray:
    """
    Transforms Delay-Doppler channel grid back to Time-Frequency domain.
    Matches MATLAB:
      1. ifftshift on both axes
      2. Doppler -> Time via IFFT along axis -1 (symbols)
      3. Delay -> Frequency via FFT along axis -2 (subcarriers)
    """
    grid_unshifted = np.fft.ifftshift(H_dd, axes=(-2, -1))
    grid_delay_time = np.fft.ifft(grid_unshifted, axis=-1)
    grid_tf = np.fft.fft(grid_delay_time, axis=-2)
    return grid_tf


def fda_blend_channels(
    H_content: np.ndarray,
    H_style: np.ndarray,
    win_h_px: int = 13,
    win_w_px: int = 3,
    window_type: str = "tukey_outward",
    taper_h: int = 2,
    taper_w: int = 1,
    scale_mode: str = "none",
) -> np.ndarray:
    """
    Applies Delay-Doppler Fourier Domain Adaptation (FDA) to swap low-frequency amplitudes.

    Parameters
    ----------
    H_content : np.ndarray
        Base channel providing phase (sample details) and outer high-frequency components.
        Shape: (..., 132, 14), complex.
    H_style : np.ndarray
        Channel providing low-frequency spectral amplitude (macro-domain style: Doppler/delay).
        Can be (..., 132, 14) or a 2D prototype (132, 14).
    win_h_px : int
        Window height along delay (default 13).
    win_w_px : int
        Window width along Doppler (default 3 or 5).
    window_type : str
        'hard', 'tukey_outward', 'tukey_inward', or 'supergaussian'.
    taper_h : int
        Taper rows for smooth window (default 2).
    taper_w : int
        Taper cols for smooth window (default 1).
    scale_mode : str
        'none', 'rms', 'minmax', or 'standardize'.

    Returns
    -------
    H_adapted : np.ndarray of shape H_content.shape, complex.
    """
    grid_h = H_content.shape[-2]
    grid_w = H_content.shape[-1]

    # Pre-scaling if requested
    if scale_mode == "rms":
        p_c = np.sqrt(np.mean(np.abs(H_content) ** 2, axis=(-2, -1), keepdims=True) + 1e-12)
        p_s = np.sqrt(np.mean(np.abs(H_style) ** 2, axis=(-2, -1), keepdims=True) + 1e-12)
        H_c_norm = H_content / p_c
        H_s_norm = H_style / p_s
    else:
        H_c_norm = H_content
        H_s_norm = H_style

    # 1. Delay-Doppler domain transform
    dd_c = tf_to_delay_doppler(H_c_norm)
    if H_s_norm.ndim == 2 and H_c_norm.ndim == 3:
        # H_style is a precomputed 2D prototype
        if np.iscomplexobj(H_s_norm):
            dd_s = tf_to_delay_doppler(H_s_norm)
            amp_s = np.abs(dd_s)
        else:
            amp_s = H_s_norm  # Already amplitude spectrum
    else:
        dd_s = tf_to_delay_doppler(H_s_norm)
        amp_s = np.abs(dd_s)

    amp_c = np.abs(dd_c)
    phase_c = np.angle(dd_c)

    # 2. Generate mask
    mask = create_fda_window(
        grid_h=grid_h,
        grid_w=grid_w,
        win_h_px=win_h_px,
        win_w_px=win_w_px,
        window_type=window_type,
        taper_h=taper_h,
        taper_w=taper_w,
    )

    # 3. Blend amplitudes: inner region M comes from style, outer (1-M) from content
    amp_mixed = mask * amp_s + (1.0 - mask) * amp_c

    # 4. Reconstruct complex Delay-Doppler with content phase
    dd_mixed = amp_mixed * np.exp(1j * phase_c)

    # 5. Transform back to Time-Frequency domain
    H_adapted = delay_doppler_to_tf(dd_mixed)

    # De-scale if RMS normalization was applied
    if scale_mode == "rms":
        H_adapted = H_adapted * p_c

    return H_adapted


def compute_source_prototype_spectrum(source_dataset_path: str, max_samples: int = 256) -> np.ndarray:
    """
    Computes the mean Delay-Doppler amplitude spectrum across samples from a source dataset.
    This prototype represents the macro-spectral fingerprint of the source domain.

    Parameters
    ----------
    source_dataset_path : str
        Path to a source .mat file or directory containing source dataset files.
    max_samples : int
        Maximum number of samples to average over (default 256).

    Returns
    -------
    mean_amp : np.ndarray of shape (132, 14), float32.
    """
    if os.path.isdir(source_dataset_path):
        # Look for a representative .mat file in SNR subfolders or root
        candidates = []
        for root, _, files in os.walk(source_dataset_path):
            for f in files:
                if f.lower().endswith('.mat'):
                    candidates.append(os.path.join(root, f))
        if not candidates:
            raise FileNotFoundError(f"No .mat files found in source directory: {source_dataset_path}")
        # Prefer higher SNR or first available
        candidates.sort(key=lambda x: (
            0 if "15db" in x.lower() or "20db" in x.lower() else (1 if "10db" in x.lower() else 2),
            x
        ))
        mat_path = candidates[0]
    else:
        mat_path = source_dataset_path

    # Load channel
    try:
        data = sio.loadmat(mat_path)
    except Exception:
        with h5py.File(mat_path, 'r') as hf:
            data = {}
            for k in hf.keys():
                v = hf[k][()]
                if isinstance(v, np.ndarray) and v.dtype.names is not None:
                    if 'real' in v.dtype.names and 'imag' in v.dtype.names:
                        v = v['real'] + 1j * v['imag']
                data[k] = v

    # Find clean or perfect channel
    H_src = None
    for key in ['H_perfect', 'H_clean', 'H_perf', 'H_true', 'H_li', 'H_LI', 'H_prac']:
        if key in data:
            H_src = data[key]
            break

    if H_src is None:
        raise KeyError(f"Could not find valid channel matrix in source file '{mat_path}'")

    # Standardize dimensions to (N, 132, 14)
    if H_src.shape[0] == 14 and H_src.shape[1] == 132:
        H_src = np.transpose(H_src, (2, 1, 0))  # (14, 132, N) -> (N, 132, 14)
    elif H_src.shape[0] == 132 and H_src.shape[1] == 14:
        H_src = np.transpose(H_src, (2, 0, 1))  # (132, 14, N) -> (N, 132, 14)
    elif H_src.ndim == 2:
        if H_src.shape == (132, 14):
            H_src = H_src[None, ...]
        elif H_src.shape == (14, 132):
            H_src = H_src.T[None, ...]

    if H_src.shape[0] > max_samples:
        H_src = H_src[:max_samples]

    # Transform to Delay-Doppler and compute average amplitude
    dd_src = tf_to_delay_doppler(H_src)
    mean_amp = np.mean(np.abs(dd_src), axis=0).astype(np.float32)

    return mean_amp


def apply_pilot_consistency_projection(
    H_est: np.ndarray,
    H_pilots: np.ndarray,
    pilot_rows: np.ndarray,
    pilot_cols: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blends the neural network / FDA output with measured LS pilots at known pilot subcarrier coordinates:
    H_final(pilot_coords) = (1 - alpha) * H_est(pilot_coords) + alpha * H_measured(pilot_coords)

    Parameters
    ----------
    H_est : np.ndarray
        Estimated 2D channel of shape (N, 132, 14).
    H_pilots : np.ndarray
        Measured LS channel at pilots of shape (N, num_pilots) or (num_pilots, N).
    pilot_rows : np.ndarray
        0-based subcarrier indices of pilots (length num_pilots).
    pilot_cols : np.ndarray
        0-based OFDM symbol indices of pilots (length num_pilots).
    alpha : float
        Blending weight in [0, 1].
        alpha = 0.0: Keep pure model output.
        alpha = 1.0: Strictly overwrite with measured LS pilots.
        alpha = 0.5: Equal balance.

    Returns
    -------
    H_out : np.ndarray of shape (N, 132, 14).
    """
    if alpha <= 0.0:
        return H_est

    alpha = min(1.0, max(0.0, float(alpha)))
    H_out = np.copy(H_est)
    N = H_est.shape[0]

    if H_pilots.shape[0] != N and H_pilots.shape[1] == N:
        H_pilots = H_pilots.T

    for i in range(N):
        curr_est_pilots = H_out[i, pilot_rows, pilot_cols]
        meas_pilots = H_pilots[i, :]
        blended = (1.0 - alpha) * curr_est_pilots + alpha * meas_pilots
        H_out[i, pilot_rows, pilot_cols] = blended

    return H_out
