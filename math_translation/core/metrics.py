"""
Evaluation Metrics for Channel Estimation
Computes MMSE, NMSE (linear and dB), and Complex/Magnitude SSIM.
"""

import numpy as np
import scipy.ndimage


def compute_mmse(H_pred, H_true):
    """Mean-squared error between predicted and true complex channels."""
    return float(np.mean(np.abs(H_pred - H_true) ** 2))


def compute_nmse(H_pred, H_true):
    """Normalized Mean-squared error (averaged over samples)."""
    num = np.mean(np.abs(H_pred - H_true) ** 2, axis=(1, 2))
    denom = np.mean(np.abs(H_true) ** 2, axis=(1, 2))
    return float(np.mean(num / (denom + 1e-30)))


def compute_nmse_db(H_pred, H_true):
    """NMSE in decibels (dB)."""
    return 10.0 * np.log10(compute_nmse(H_pred, H_true) + 1e-30)


def compute_ssim_batch(H_pred, H_true):
    """
    Computes Structural Similarity Index Measure (SSIM) on channel magnitudes.
    Compatible across NumPy 1.x and 2.x using SciPy gaussian filtering.
    """
    mag_pred = np.abs(H_pred).astype(np.float32)
    mag_true = np.abs(H_true).astype(np.float32)

    mn = mag_true.min(axis=(1, 2), keepdims=True)
    mx = mag_true.max(axis=(1, 2), keepdims=True)
    scale = np.clip(mx - mn, 1e-8, None)
    mag_pred_n = np.clip((mag_pred - mn) / scale, 0.0, 1.0)
    mag_true_n = (mag_true - mn) / scale

    ssim_list = []
    C1 = 0.0001
    C2 = 0.0009

    for i in range(mag_pred_n.shape[0]):
        x = mag_pred_n[i]
        y = mag_true_n[i]

        mu_x = scipy.ndimage.gaussian_filter(x, sigma=1.5, truncate=3.5)
        mu_y = scipy.ndimage.gaussian_filter(y, sigma=1.5, truncate=3.5)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = scipy.ndimage.gaussian_filter(x ** 2, sigma=1.5, truncate=3.5) - mu_x_sq
        sigma_y_sq = scipy.ndimage.gaussian_filter(y ** 2, sigma=1.5, truncate=3.5) - mu_y_sq
        sigma_xy = scipy.ndimage.gaussian_filter(x * y, sigma=1.5, truncate=3.5) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denom = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

        ssim_val = np.mean(num / (denom + 1e-30))
        ssim_list.append(ssim_val)

    return float(np.mean(ssim_list))

