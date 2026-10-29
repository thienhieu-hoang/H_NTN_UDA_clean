"""
Blind Pilot Statistical Shift Estimator
Extracts physical Delay Dilation (beta_tau) and bulk Doppler Shift (delta_nu)
purely from sparse pilot observations without any target channel labels.
"""

import numpy as np


def compute_pilot_freq_autocorr(H_pilots, pilot_rows, pilot_cols, max_lag=30):
    """
    Computes the sample frequency autocorrelation function R_f(Delta_k) from pilots.
    
    Args:
        H_pilots: (N, P) complex array of pilot observations
        pilot_rows: (P,) array of 0-based subcarrier indices
        pilot_cols: (P,) array of 0-based symbol indices
        max_lag: maximum subcarrier lag to evaluate
        
    Returns:
        lags: (L,) array of evaluated subcarrier lags
        R_f: (L,) normalized autocorrelation values |R_f(Delta_k)|
    """
    unique_cols = np.unique(pilot_cols)
    N = H_pilots.shape[0]

    # Pre-organize pilot subcarriers and indices for each pilot symbol
    symbol_pilot_data = []
    for col in unique_cols:
        col_mask = (pilot_cols == col)
        subcarriers = pilot_rows[col_mask]
        pilot_indices = np.where(col_mask)[0]
        # Sort by subcarrier
        sort_idx = np.argsort(subcarriers)
        symbol_pilot_data.append((subcarriers[sort_idx], pilot_indices[sort_idx]))

    # Compute correlation across all pilot pairs
    lag_corr_sum = {}
    lag_corr_count = {}

    for subcarriers, p_idx in symbol_pilot_data:
        num_p = len(subcarriers)
        for i in range(num_p):
            for j in range(i, num_p):
                lag = int(subcarriers[j] - subcarriers[i])
                if lag > max_lag:
                    continue
                # Product: H(k + Delta_k) * conj(H(k))
                val = np.mean(H_pilots[:, p_idx[j]] * np.conj(H_pilots[:, p_idx[i]]))
                if lag not in lag_corr_sum:
                    lag_corr_sum[lag] = 0.0
                    lag_corr_count[lag] = 0
                lag_corr_sum[lag] += val
                lag_corr_count[lag] += 1

    sorted_lags = sorted(lag_corr_sum.keys())
    R_f = np.array([np.abs(lag_corr_sum[l] / lag_corr_count[l]) for l in sorted_lags])
    
    # Normalize by lag 0
    if len(R_f) > 0 and R_f[0] > 1e-30:
        R_f = R_f / R_f[0]

    return np.array(sorted_lags), R_f


def estimate_coherence_bandwidth_lag(lags, R_f, threshold=0.5):
    """
    Finds the subcarrier lag Delta_k where frequency correlation drops below threshold.
    Uses linear interpolation between discrete lags.
    """
    if len(R_f) == 0:
        return 1.0
    below_thresh = np.where(R_f < threshold)[0]
    if len(below_thresh) == 0:
        # Correlation stays high; fit exponential decay R_f(k) = exp(-k / k_0)
        # k_0 = -k / ln(R_f(k))
        valid_idx = np.where(R_f > 0.01)[0]
        if len(valid_idx) > 1:
            k_last = lags[valid_idx[-1]]
            r_last = np.clip(R_f[valid_idx[-1]], 1e-5, 0.999)
            k_0 = -k_last / np.log(r_last)
            return float(k_0 * np.log(1.0 / threshold))
        return float(lags[-1])

    idx = below_thresh[0]
    if idx == 0:
        return float(lags[0])
    
    # Linear interpolation between idx-1 and idx
    k1, k2 = lags[idx - 1], lags[idx]
    r1, r2 = R_f[idx - 1], R_f[idx]
    if abs(r2 - r1) < 1e-12:
        return float(k1)
    k_star = k1 + (threshold - r1) * (k2 - k1) / (r2 - r1)
    return float(k_star)


def estimate_delay_dilation_autocorr(H_src_pilots, H_tgt_pilots, pilot_rows, pilot_cols, max_lag=30, threshold=0.5):
    """
    Estimates the delay dilation factor beta_tau = Delta_k_src / Delta_k_tgt
    blindly from source and target pilots.
    """
    lags_src, R_src = compute_pilot_freq_autocorr(H_src_pilots, pilot_rows, pilot_cols, max_lag)
    lags_tgt, R_tgt = compute_pilot_freq_autocorr(H_tgt_pilots, pilot_rows, pilot_cols, max_lag)

    k_src = estimate_coherence_bandwidth_lag(lags_src, R_src, threshold)
    k_tgt = estimate_coherence_bandwidth_lag(lags_tgt, R_tgt, threshold)

    beta_tau = k_src / max(k_tgt, 0.1)
    return float(beta_tau), (lags_src, R_src, k_src), (lags_tgt, R_tgt, k_tgt)


def estimate_bulk_doppler_shift(H_src_pilots, H_tgt_pilots, pilot_rows, pilot_cols, T_sym=3.5714e-5):
    """
    Estimates relative bulk Doppler shift Delta_nu (in Hz) between target and source pilots.
    Computes temporal correlation phase across pilot symbols (symbols 2 and 11, Delta_l = 9).
    
    Args:
        T_sym: OFDM symbol duration in seconds (for 30 kHz SCS, T_sym ~ 35.71 us)
    """
    unique_cols = np.unique(pilot_cols)
    if len(unique_cols) < 2:
        return 0.0, 0.0, 0.0

    col1, col2 = unique_cols[0], unique_cols[1]
    delta_l = col2 - col1
    delta_t = delta_l * T_sym

    mask1 = (pilot_cols == col1)
    mask2 = (pilot_cols == col2)
    rows1 = pilot_rows[mask1]
    rows2 = pilot_rows[mask2]

    # Find common subcarriers present in both symbols
    common_rows = np.intersect1d(rows1, rows2)
    if len(common_rows) == 0:
        return 0.0, 0.0, 0.0

    idx1 = [np.where((pilot_cols == col1) & (pilot_rows == r))[0][0] for r in common_rows]
    idx2 = [np.where((pilot_cols == col2) & (pilot_rows == r))[0][0] for r in common_rows]

    # Average temporal correlation
    R_t_src = np.mean(H_src_pilots[:, idx2] * np.conj(H_src_pilots[:, idx1]))
    R_t_tgt = np.mean(H_tgt_pilots[:, idx2] * np.conj(H_tgt_pilots[:, idx1]))

    phase_src = np.angle(R_t_src)
    phase_tgt = np.angle(R_t_tgt)

    phase_diff = np.angle(np.exp(1j * (phase_tgt - phase_src)))
    delta_nu = phase_diff / (2.0 * np.pi * delta_t)

    return float(delta_nu), float(phase_src), float(phase_tgt)


def estimate_delay_doppler_moments(H_grid):
    """
    Computes 2D Delay-Doppler statistical moments (center of mass and spread)
    for a channel grid (N, 132, 14) or (132, 14).
    """
    if H_grid.ndim == 2:
        H_grid = H_grid[np.newaxis, ...]

    # 2D FFT: IFFT along subcarriers (delay), FFT along symbols (Doppler)
    H_dd = np.fft.fftshift(np.fft.fft(np.fft.ifft(H_grid, axis=1), axis=2), axes=(1, 2))
    power = np.abs(H_dd) ** 2
    avg_power = np.mean(power, axis=0)  # (132, 14)

    total_p = np.sum(avg_power) + 1e-30
    norm_p = avg_power / total_p

    N_sc, N_sym = avg_power.shape
    tau_grid = np.arange(N_sc) - N_sc // 2
    nu_grid = np.arange(N_sym) - N_sym // 2

    # Marginal distributions
    p_tau = np.sum(norm_p, axis=1)  # (N_sc,)
    p_nu = np.sum(norm_p, axis=0)   # (N_sym,)

    mu_tau = np.sum(tau_grid * p_tau)
    sigma_tau = np.sqrt(np.maximum(0.0, np.sum(((tau_grid - mu_tau) ** 2) * p_tau)))

    mu_nu = np.sum(nu_grid * p_nu)
    sigma_nu = np.sqrt(np.maximum(0.0, np.sum(((nu_grid - mu_nu) ** 2) * p_nu)))

    return {
        'mu_tau': float(mu_tau),
        'sigma_tau': float(sigma_tau),
        'mu_nu': float(mu_nu),
        'sigma_nu': float(sigma_nu),
        'power_dd': avg_power
    }
