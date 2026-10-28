"""
========================================================================================
FDA Inference Strategies Library for Wireless Channel Estimation
========================================================================================
Defines clean, modular execution strategies for all FDA inference paradigms:
- Approach 2.1: Dual FDA with Source Input (Pre-FDA disguise + Post-FDA + PCP)
- Approach 2.2: Dual FDA with Source Label (Clean source center disguise)
- Approach 2.3: Raw Baseline (Straight forward inference without FDA)
- Approach 2.4: Post-FDA Only (Raw forward pass + Post-FDA restoration + PCP)
- Approach 2.5: Iterative Self-Refinement (2-Pass FDA using Pass 1 clean center)
- Approach 3.1: Model 2 Straight Inference (Trained on pseudo-dataset + PCP)
- Approach 3.2: Model 2 Post-FDA Bias Correction (Post-FDA restoration + PCP)
- Approach 3.4: Teacher-Student Cross-Refinement (Model 2 Macro + Model 1 Clean Phase)
========================================================================================
"""

import os
import numpy as np
import onnxruntime as ort

from fda_window_utils import (
    fda_blend_channels,
    apply_pilot_consistency_projection,
)


# ============================================================================
# NORMALIZATION UTILITIES
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
    x_min_b = np.asarray(x_min).reshape(reshape_dims)
    x_max_b = np.asarray(x_max).reshape(reshape_dims)
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
    x_rms = np.sqrt(np.mean(x_reshaped ** 2, axis=(1, 2)))  # (N,)
    x_rms = np.clip(x_rms, 1e-30, None)
    reshape_dims = [N] + [1] * (x.ndim - 1)
    x_rms_b = x_rms.reshape(reshape_dims)
    x_scaled = x / x_rms_b
    return x_scaled, x_rms


def de_rms(x_normd, x_rms):
    N = x_normd.shape[0]
    reshape_dims = [N] + [1] * (x_normd.ndim - 1)  # For (N, 132, 14, 2) -> (N, 1, 1, 1)
    x_rms_b = np.asarray(x_rms).reshape(reshape_dims)
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
    x_mean_b = np.asarray(x_mean).reshape(reshape_dims)
    x_std_b = np.asarray(x_std).reshape(reshape_dims)
    return x_normd * x_std_b + x_mean_b


# ============================================================================
# UNIFIED ONNX FORWARD INFERENCE ENGINE
# ============================================================================

def predict_channel_onnx(
    H_in_grid: np.ndarray,
    onnx_path: str,
    is_sequence_model: bool,
    model_name_lower: str,
    norm_mode: str,
    pilot_rows: np.ndarray = None,
    pilot_cols: np.ndarray = None,
    **kwargs
) -> np.ndarray:
    """
    Executes a complete ONNX forward inference pass for either sequence or grid models.

    Parameters
    ----------
    H_in_grid : np.ndarray
        Input channel grid of shape (N, 132, 14), complex128.
    onnx_path : str
        Path to the best_model.onnx file.
    is_sequence_model : bool
        True if model expects pilot sequence (e.g. LS_Attention), False if grid (LI_DnCNN).
    model_name_lower : str
        Lowercased model name (for residual handling like DnCNN).
    norm_mode : str
        'rms', 'standardize', or 'minmax'.
    pilot_rows, pilot_cols : np.ndarray
        Coordinates of pilot positions in (132, 14) grid.

    Returns
    -------
    H_pred : np.ndarray of shape (N, 132, 14), complex128.
    """
    N_samples = H_in_grid.shape[0]

    # Step 1: Format model input
    if is_sequence_model:
        # Extract 88 pilots from grid
        H_pilots = np.zeros((N_samples, len(pilot_rows)), dtype=np.complex128)
        for i in range(N_samples):
            H_pilots[i, :] = H_in_grid[i, pilot_rows, pilot_cols]
        # Stack Real and Imag: (N, 88, 2)
        x = np.stack([H_pilots.real, H_pilots.imag], axis=-1).astype(np.float32)
    else:
        # Stack 2D Grid Real and Imag: (N, 132, 14, 2)
        x = np.stack([H_in_grid.real, H_in_grid.imag], axis=-1).astype(np.float32)

    # Step 2: Normalization Scaling
    if norm_mode == "rms":
        x_scaled, x_rms = rms_scaler(x)
    elif norm_mode == "standardize":
        x_scaled, x_mean, x_std = standardize_scaler(x)
    else:
        x_scaled, x_min, x_max = minmax_scaler(x, lower_range=-1)

    # Step 3: Run ONNX Session
    sess = ort.InferenceSession(onnx_path)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    in_shape = sess.get_inputs()[0].shape

    # Adaptive NCHW permutation mapping for 4D grid models
    input_perm = None
    if len(in_shape) == 4 and not is_sequence_model:
        perm = []
        for dim in in_shape:
            if dim == 2:
                perm.append(3)
            elif dim == 132:
                perm.append(1)
            elif dim == 14:
                perm.append(2)
            else:
                perm.append(0)
        input_perm = tuple(perm)

    y_pred_list = []
    for idx in range(N_samples):
        x_sample = x_scaled[idx:idx + 1].astype(np.float32)
        if input_perm is not None:
            x_sample = np.transpose(x_sample, input_perm)
        y_out = sess.run([out_name], {in_name: x_sample})[0]

        # Align output to (132, 14, 2)
        y_sq = np.squeeze(y_out)
        sq_shape = list(y_sq.shape)
        if 132 in sq_shape and 14 in sq_shape and 2 in sq_shape:
            idx_132 = sq_shape.index(132)
            idx_14 = sq_shape.index(14)
            idx_2 = sq_shape.index(2)
            y_aligned = np.transpose(y_sq, (idx_132, idx_14, idx_2))
        else:
            y_aligned = y_sq.reshape(132, 14, 2)

        # Residual add-back for DnCNN
        if not is_sequence_model and "dncnn" in model_name_lower:
            y_aligned = x_scaled[idx] + y_aligned

        y_pred_list.append(y_aligned)

    y_pred_scaled = np.array(y_pred_list)  # (N, 132, 14, 2)

    # Step 4: Denormalization
    if norm_mode == "rms":
        x_denorm = de_rms(y_pred_scaled, x_rms)
    elif norm_mode == "standardize":
        x_denorm = de_standardize(y_pred_scaled, x_mean, x_std)
    else:
        x_denorm = de_min_max(y_pred_scaled, x_min, x_max, lower_range=-1)

    # Combine complex channel
    H_pred = x_denorm[..., 0] + 1j * x_denorm[..., 1]
    return H_pred


def _apply_pcp_helper(H_est, H_target_grid, pilot_rows, pilot_cols, alpha=0.5):
    """Helper to apply PCP using pilot coordinates."""
    if alpha <= 0.0 or pilot_rows is None or len(pilot_rows) == 0:
        return H_est
    N = H_est.shape[0]
    H_pilots = np.zeros((N, len(pilot_rows)), dtype=np.complex128)
    for i in range(N):
        H_pilots[i, :] = H_target_grid[i, pilot_rows, pilot_cols]
    return apply_pilot_consistency_projection(H_est, H_pilots, pilot_rows, pilot_cols, alpha)


# ============================================================================
# FDA INFERENCE APPROACH STRATEGIES
# ============================================================================

def approach_2_1_dual_fda(
    H_target: np.ndarray,
    onnx_path: str,
    source_ref: np.ndarray,
    window_kwargs: dict,
    model_cfg: dict,
    alpha: float = 0.5,
    **kwargs
) -> np.ndarray:
    """Approach 2.1: Dual FDA with Source Input (Current Baseline)."""
    # Step 1: Pre-FDA disguise
    H_in = fda_blend_channels(H_content=H_target, H_style=source_ref, **window_kwargs)
    # Step 2: Model forward pass
    H_pred = predict_channel_onnx(H_in, onnx_path, **model_cfg)
    # Step 3: Post-FDA restoration
    H_restored = fda_blend_channels(H_content=H_pred, H_style=H_target, **window_kwargs)
    # Step 4: Pilot consistency projection
    return _apply_pcp_helper(H_restored, H_target, model_cfg.get("pilot_rows"), model_cfg.get("pilot_cols"), alpha)


def approach_2_2_clean_source(
    H_target: np.ndarray,
    onnx_path: str,
    source_ref: np.ndarray,
    window_kwargs: dict,
    model_cfg: dict,
    alpha: float = 0.5,
    **kwargs
) -> np.ndarray:
    """Approach 2.2: Dual FDA with Source Label (100% clean source center)."""
    # Step 1: Pre-FDA with clean source prototype
    H_in = fda_blend_channels(H_content=H_target, H_style=source_ref, **window_kwargs)
    # Step 2: Model forward pass
    H_pred = predict_channel_onnx(H_in, onnx_path, **model_cfg)
    # Step 3: Post-FDA restoration
    H_restored = fda_blend_channels(H_content=H_pred, H_style=H_target, **window_kwargs)
    # Step 4: Pilot anchoring
    return _apply_pcp_helper(H_restored, H_target, model_cfg.get("pilot_rows"), model_cfg.get("pilot_cols"), alpha)


def approach_2_3_raw_baseline(
    H_target: np.ndarray,
    onnx_path: str,
    model_cfg: dict,
    alpha: float = 0.5,
    **kwargs
) -> np.ndarray:
    """Approach 2.3: Raw Baseline (Straight forward inference without FDA)."""
    H_pred = predict_channel_onnx(H_target, onnx_path, **model_cfg)
    return _apply_pcp_helper(H_pred, H_target, model_cfg.get("pilot_rows"), model_cfg.get("pilot_cols"), alpha)


def approach_2_4_post_only(
    H_target: np.ndarray,
    onnx_path: str,
    window_kwargs: dict,
    model_cfg: dict,
    alpha: float = 0.5,
    **kwargs
) -> np.ndarray:
    """Approach 2.4: Post-FDA Only (Raw input forward pass + Post-FDA restoration)."""
    H_pred = predict_channel_onnx(H_target, onnx_path, **model_cfg)
    H_restored = fda_blend_channels(H_content=H_pred, H_style=H_target, **window_kwargs)
    return _apply_pcp_helper(H_restored, H_target, model_cfg.get("pilot_rows"), model_cfg.get("pilot_cols"), alpha)


def approach_2_5_self_refine(
    H_target: np.ndarray,
    onnx_path: str,
    source_ref: np.ndarray,
    window_kwargs: dict,
    model_cfg: dict,
    alpha: float = 0.5,
    **kwargs
) -> np.ndarray:
    """
    Approach 2.5: Iterative Self-Refinement (2-Pass FDA).
    Pass 1: Runs Approach 2.1 to generate initial denoised channel H_pass1.
    Pass 2: Uses H_pass1's clean center as the style donor for a second pass.
    """
    # Pass 1: Initial estimate via Approach 2.1
    H_pass1 = approach_2_1_dual_fda(H_target, onnx_path, source_ref, window_kwargs, model_cfg, alpha)

    # Pass 2: Denoised style donor
    H_in2 = fda_blend_channels(H_content=H_target, H_style=H_pass1, **window_kwargs)
    H_pred2 = predict_channel_onnx(H_in2, onnx_path, **model_cfg)
    H_restored2 = fda_blend_channels(H_content=H_pred2, H_style=H_pass1, **window_kwargs)
    return _apply_pcp_helper(H_restored2, H_target, model_cfg.get("pilot_rows"), model_cfg.get("pilot_cols"), alpha)


def approach_3_1_model2_raw(
    H_target: np.ndarray,
    onnx_path: str,
    model_cfg: dict,
    alpha: float = 0.5,
    **kwargs
) -> np.ndarray:
    """Approach 3.1: Model 2 Straight Inference (Trained on pseudo-dataset)."""
    H_pred = predict_channel_onnx(H_target, onnx_path, **model_cfg)
    return _apply_pcp_helper(H_pred, H_target, model_cfg.get("pilot_rows"), model_cfg.get("pilot_cols"), alpha)


def approach_3_2_model2_post(
    H_target: np.ndarray,
    onnx_path: str,
    window_kwargs: dict,
    model_cfg: dict,
    alpha: float = 0.5,
    **kwargs
) -> np.ndarray:
    """Approach 3.2: Model 2 Post-FDA Bias Correction."""
    H_pred = predict_channel_onnx(H_target, onnx_path, **model_cfg)
    H_restored = fda_blend_channels(H_content=H_pred, H_style=H_target, **window_kwargs)
    return _apply_pcp_helper(H_restored, H_target, model_cfg.get("pilot_rows"), model_cfg.get("pilot_cols"), alpha)


def approach_3_3_model2_dual(
    H_target: np.ndarray,
    onnx_path: str,
    source_ref: np.ndarray,
    window_kwargs: dict,
    model_cfg: dict,
    alpha: float = 0.5,
    **kwargs
) -> np.ndarray:
    """Approach 3.3: Model 2 Pre-FDA Alignment to Training Pseudo Prototype + Post-FDA Bias Correction."""
    # Step 1: Pre-FDA input alignment using training reference prototype (source_ref)
    H_in = fda_blend_channels(H_content=H_target, H_style=source_ref, **window_kwargs)
    # Step 2: Model forward pass (model2)
    H_pred = predict_channel_onnx(H_in, onnx_path, **model_cfg)
    # Step 3: Post-FDA restoration with physical target input
    H_restored = fda_blend_channels(H_content=H_pred, H_style=H_target, **window_kwargs)
    # Step 4: Pilot consistency projection
    return _apply_pcp_helper(H_restored, H_target, model_cfg.get("pilot_rows"), model_cfg.get("pilot_cols"), alpha)


def approach_3_4_ensemble(
    H_target: np.ndarray,
    onnx_path: str,
    window_kwargs: dict,
    model_cfg: dict,
    onnx_model2_path: str = None,
    alpha: float = 0.5,
    **kwargs
) -> np.ndarray:
    """Approach 3.4 / 2.6: Teacher-Student Ensemble (Model 2 Macro Prior + Model 1 Sharp Phase)."""
    if onnx_model2_path is None or not os.path.isfile(onnx_model2_path):
        # Fallback to Approach 2.1 if model2 is not specified
        return approach_2_1_dual_fda(H_target, onnx_path, kwargs.get("source_ref"), window_kwargs, model_cfg, alpha)

    # Step 1: Model 2 initial guess
    H_m2 = predict_channel_onnx(H_target, onnx_model2_path, **model_cfg)
    # Step 2: Disguise Model 1 with Model 2's clean target Doppler
    H_in1 = fda_blend_channels(H_content=H_target, H_style=H_m2, **window_kwargs)
    H_pred1 = predict_channel_onnx(H_in1, onnx_path, **model_cfg)
    # Step 3: Post-FDA & PCP
    H_restored = fda_blend_channels(H_content=H_pred1, H_style=H_target, **window_kwargs)
    return _apply_pcp_helper(H_restored, H_target, model_cfg.get("pilot_rows"), model_cfg.get("pilot_cols"), alpha)


# ============================================================================
# STRATEGY DISPATCHER REGISTRY
# ============================================================================

APPROACH_REGISTRY = {
    "2.1": approach_2_1_dual_fda,
    "app2_1": approach_2_1_dual_fda,
    "dual": approach_2_1_dual_fda,

    "2.2": approach_2_2_clean_source,
    "app2_2": approach_2_2_clean_source,

    "2.3": approach_2_3_raw_baseline,
    "app2_3": approach_2_3_raw_baseline,
    "raw": approach_2_3_raw_baseline,

    "2.4": approach_2_4_post_only,
    "app2_4": approach_2_4_post_only,
    "post_only": approach_2_4_post_only,

    "2.5": approach_2_5_self_refine,
    "app2_5": approach_2_5_self_refine,
    "self_refine": approach_2_5_self_refine,

    "2.6": approach_3_4_ensemble,
    "app2_6": approach_3_4_ensemble,

    "3.1": approach_3_1_model2_raw,
    "app3_1": approach_3_1_model2_raw,

    "3.2": approach_3_2_model2_post,
    "app3_2": approach_3_2_model2_post,

    "3.3": approach_3_3_model2_dual,
    "app3_3": approach_3_3_model2_dual,
    "model2_dual": approach_3_3_model2_dual,

    "3.4": approach_3_4_ensemble,
    "app3_4": approach_3_4_ensemble,
    "ensemble": approach_3_4_ensemble,
}


def get_approach_executor(approach_id: str):
    """Returns the approach strategy function corresponding to approach_id."""
    key = str(approach_id).lower().strip()
    if key not in APPROACH_REGISTRY:
        raise ValueError(
            f"Unknown approach ID: '{approach_id}'. "
            f"Supported approaches: {sorted(list(APPROACH_REGISTRY.keys()))}"
        )
    return APPROACH_REGISTRY[key]
