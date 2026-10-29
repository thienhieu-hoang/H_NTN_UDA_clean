"""
Physical Channel Warping Operators
Implements physical Delay Dilation (continuous stretching in Delay domain /
coherence bandwidth compression in Frequency domain), Doppler phase modulation,
inverse transformations, and pilot anchoring.
"""

import numpy as np
import scipy.interpolate


def dilate_delay_channel_grid(H_grid, beta_tau):
    """
    Physically dilates or compresses the delay spread of a channel grid (N, 132, 14)
    by a factor beta_tau using exact Delay-domain coordinate transformation.
    
    Physics:
        h(tau, t) = IFFT_f { H(f, t) }
        h_warped(tau, t) = (1 / sqrt(beta_tau)) * h(tau / beta_tau, t)
        H_warped(f, t) = FFT_tau { h_warped(tau, t) }
        
    Args:
        H_grid: (N, N_sc, N_sym) complex channel tensor (e.g. N x 132 x 14)
        beta_tau: delay dilation factor (> 1 stretches delay spread; < 1 compresses)
        
    Returns:
        H_warped: (N, N_sc, N_sym) warped complex channel tensor
    """
    beta_tau = float(np.clip(beta_tau, 0.05, 20.0))
    if abs(beta_tau - 1.0) < 1e-4:
        return H_grid.copy()

    orig_shape = H_grid.shape
    if H_grid.ndim == 2:
        H_grid = H_grid[np.newaxis, ...]

    N, N_sc, N_sym = H_grid.shape

    # 1. Transform frequency subcarriers to delay domain via IFFT
    # Impulse response h has shape (N, N_sc, N_sym)
    h_delay = np.fft.ifft(H_grid, axis=1)

    # 2. Coordinate transformation in Delay domain
    # In NTN / wireless, channel delay response is causal (tau >= 0) with zero-padding
    tau_orig = np.arange(N_sc, dtype=np.float64)
    tau_src = tau_orig / float(beta_tau)

    # Valid interpolation points within [0, N_sc - 1]
    i0 = np.floor(tau_src).astype(int)
    i1 = i0 + 1
    frac = (tau_src - i0)[np.newaxis, :, np.newaxis]

    mask_both = (i1 < N_sc)
    mask_i0_only = (i0 < N_sc) & (~mask_both)

    h_warped = np.zeros_like(h_delay)
    if np.any(mask_both):
        idx_b0 = i0[mask_both]
        idx_b1 = i1[mask_both]
        f_b = frac[:, mask_both, :]
        h_warped[:, mask_both, :] = (1.0 - f_b) * h_delay[:, idx_b0, :] + f_b * h_delay[:, idx_b1, :]

    if np.any(mask_i0_only):
        idx_only = i0[mask_i0_only]
        h_warped[:, mask_i0_only, :] = h_delay[:, idx_only, :]

    # Preserve power: scale by energy factor
    power_orig = np.mean(np.abs(H_grid) ** 2, axis=(1, 2), keepdims=True)
    
    # 3. Transform back to frequency domain via FFT
    H_warped = np.fft.fft(h_warped, axis=1)
    
    power_warped = np.mean(np.abs(H_warped) ** 2, axis=(1, 2), keepdims=True) + 1e-30
    H_warped = H_warped * np.sqrt(power_orig / power_warped)

    if len(orig_shape) == 2:
        return H_warped[0]
    return H_warped


def modulate_doppler_channel_grid(H_grid, delta_nu, T_sym=3.5714e-5):
    """
    Modulates a channel grid (N, 132, 14) by a bulk Doppler shift delta_nu (Hz).
    
    Physics:
        H_mod(f, t) = H(f, t) * exp(j * 2*pi * delta_nu * t * T_sym)
    """
    if abs(delta_nu) < 1e-2:
        return H_grid.copy()

    orig_shape = H_grid.shape
    if H_grid.ndim == 2:
        H_grid = H_grid[np.newaxis, ...]

    N, N_sc, N_sym = H_grid.shape
    t_symbols = np.arange(N_sym, dtype=np.float64)
    phase_ramp = np.exp(1j * 2.0 * np.pi * delta_nu * t_symbols * T_sym)  # (N_sym,)

    H_mod = H_grid * phase_ramp[np.newaxis, np.newaxis, :]

    if len(orig_shape) == 2:
        return H_mod[0]
    return H_mod


def warp_source_to_target(H_src, beta_tau, delta_nu, T_sym=3.5714e-5):
    """
    Full forward mathematical translation from Source domain to Target domain:
    Applies Delay Dilation (beta_tau) followed by Doppler Modulation (delta_nu).
    """
    H_dilated = dilate_delay_channel_grid(H_src, beta_tau)
    H_target = modulate_doppler_channel_grid(H_dilated, delta_nu, T_sym)
    return H_target


def inverse_warp_target_to_source(H_tgt, beta_tau, delta_nu, T_sym=3.5714e-5):
    """
    Inverse mathematical translation from Target domain back to Source domain:
    Demodulates Doppler (-delta_nu) and compresses Delay spread (1 / beta_tau).
    """
    H_demod = modulate_doppler_channel_grid(H_tgt, -delta_nu, T_sym)
    H_src_like = dilate_delay_channel_grid(H_demod, 1.0 / beta_tau)
    return H_src_like


def warp_pilots_vector(H_pilots, pilot_rows, pilot_cols, beta_tau, delta_nu, forward=True, T_sym=3.5714e-5, n_sc=132, n_sym=14):
    """
    Warps pilot observations between source and target channel geometries:
    1. Reconstructs smooth intermediate channel grid via 2D interpolation.
    2. Applies physical Delay Dilation and Doppler phase modulation in grid domain.
    3. Samples warped channel at the exact DM-RS pilot positions (r_p, c_p).
    """
    N = H_pilots.shape[0]
    eff_beta = beta_tau if forward else (1.0 / beta_tau)
    eff_dnu = delta_nu if forward else (-delta_nu)

    # 1. 2D Linear Interpolation of pilots to full grid (N, 132, 14)
    grid_r, grid_c = np.meshgrid(np.arange(n_sc), np.arange(n_sym), indexing='ij')
    points = np.column_stack([pilot_rows, pilot_cols])
    H_grid = np.zeros((N, n_sc, n_sym), dtype=np.complex128)

    for n in range(N):
        real_p = H_pilots[n].real
        imag_p = H_pilots[n].imag
        real_grid = scipy.interpolate.griddata(points, real_p, (grid_r, grid_c), method='linear')
        imag_grid = scipy.interpolate.griddata(points, imag_p, (grid_r, grid_c), method='linear')

        # Fill edge extrapolations
        if np.isnan(real_grid).any():
            real_near = scipy.interpolate.griddata(points, real_p, (grid_r, grid_c), method='nearest')
            imag_near = scipy.interpolate.griddata(points, imag_p, (grid_r, grid_c), method='nearest')
            mask = np.isnan(real_grid)
            real_grid[mask] = real_near[mask]
            imag_grid[mask] = imag_near[mask]

        H_grid[n] = real_grid + 1j * imag_grid

    # 2. Apply physical coordinate transformation
    H_warped_grid = dilate_delay_channel_grid(H_grid, eff_beta)
    if abs(eff_dnu) > 1e-2:
        H_warped_grid = modulate_doppler_channel_grid(H_warped_grid, eff_dnu, T_sym)

    # 3. Sample at exact pilot positions
    H_warped_pilots = H_warped_grid[:, pilot_rows, pilot_cols]

    # Normalize power to match original pilot power
    p_orig = np.mean(np.abs(H_pilots) ** 2, axis=1, keepdims=True)
    p_warp = np.mean(np.abs(H_warped_pilots) ** 2, axis=1, keepdims=True) + 1e-30
    H_warped_pilots = H_warped_pilots * np.sqrt(p_orig / p_warp)

    return H_warped_pilots


def anchor_pilots_into_grid(H_pred_grid, H_ls_pilots, pilot_rows, pilot_cols, blend_factor=1.0):
    """
    Substitutes / blends known target pilot measurements into the predicted channel grid
    to guarantee exact pilot fidelity and eliminate residual estimation error at pilot REs.
    
    Args:
        H_pred_grid: (N, 132, 14) predicted full channel grid
        H_ls_pilots: (N, 88) measured LS pilot values
        pilot_rows: (88,) 0-based subcarrier indices
        pilot_cols: (88,) 0-based symbol indices
        blend_factor: 1.0 = hard substitution; < 1.0 = soft blending
    """
    H_anchored = H_pred_grid.copy()
    N = H_pred_grid.shape[0]

    for p in range(len(pilot_rows)):
        r = pilot_rows[p]
        c = pilot_cols[p]
        orig_val = H_anchored[:, r, c]
        pilot_val = H_ls_pilots[:, p]
        H_anchored[:, r, c] = (1.0 - blend_factor) * orig_val + blend_factor * pilot_val

    return H_anchored
