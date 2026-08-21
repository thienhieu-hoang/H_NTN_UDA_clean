import os
import sys
import re
import shutil
import time
import numpy as np
import scipy.io
import h5py
import onnxruntime as ort

# ============================================================================
# CONFIGURATION CONSTANTS
# Edit these paths and parameters directly to run from your IDE
# ============================================================================
MODEL_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DnCNN_Attention_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps"
DATASET_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\MATLAB\sampleWiseDoppler_wGeometry_A100_2p18e9_600km_70deg_30kHz"
OUT_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\inference\DUR100__A100_2p18e9_600km_30kHz_DnCNN_ResNet_Attention"
MODEL_NAME = "best_net.onnx"
CLIP_EXTRAP = "auto"       # "auto", "true", or "false" (auto detects if "clip" is in model_dir name)
OUTPUT_KEY = "auto"        # "auto" saves variable as "H_infer", otherwise custom string
NUM_SAMPLES = 64           # Integer (e.g. 512, 64) or None to process all samples in the dataset
MODEL_TYPE = "LS"          # "auto" (process all), "LI" (only LI subfolders), or "LS" (only LS subfolders)
# ============================================================================

def extract_snr(folder_name):
    """
    Extracts the integer SNR value from a folder name.
    Matches cases like LI_-10, LI_-5, LI_0, LI_5, LI_10, LI_15, or SNR_-10dB, etc.
    """
    match = re.search(r'[-+]?\d+', folder_name)
    if match:
        return int(match.group())
    return None

def h5_to_complex(dataset_value):
    """
    Converts h5py compound dataset (real, imag) to standard complex numpy array.
    """
    if isinstance(dataset_value, np.ndarray) and dataset_value.dtype.names is not None:
        if 'real' in dataset_value.dtype.names and 'imag' in dataset_value.dtype.names:
            return dataset_value['real'] + 1j * dataset_value['imag']
    return dataset_value

def minmax_scaler(x, lower_range=-1):
    """
    Scales the input tensor between lower_range and 1 along axis 1 & 2 for each sample.
    Matches the minmaxScaler function used during training.
    """
    N = x.shape[0]
    # Flatten spatial dimensions to compute min/max per sample and channel
    x_reshaped = x.reshape(N, -1, 2)
    x_min = np.min(x_reshaped, axis=1)  # (N, 2)
    x_max = np.max(x_reshaped, axis=1)  # (N, 2)
    
    scale = np.clip(x_max - x_min, 1e-30, None)  # Avoid division by zero
    
    x_min_b = x_min[:, np.newaxis, np.newaxis, :]
    scale_b = scale[:, np.newaxis, np.newaxis, :]
    
    x_scaled = (x - x_min_b) / scale_b
    if lower_range == -1:
        x_scaled = x_scaled * 2.0 - 1.0
        
    return x_scaled, x_min, x_max

def de_min_max(x_normd, x_min, x_max, lower_range=-1):
    """
    Denormalizes the scaled tensor back to its original scale.
    Matches the deMinMax function used during training.
    """
    if lower_range == -1:
        scale = (x_max - x_min) / 2.0
        shift = (x_max + x_min) / 2.0
        return x_normd * scale[:, np.newaxis, np.newaxis, :] + shift[:, np.newaxis, np.newaxis, :]
    else:
        scale = x_max - x_min
        return x_normd * scale[:, np.newaxis, np.newaxis, :] + x_min[:, np.newaxis, np.newaxis, :]


def match_original_shape(H_est, source_mat_path):
    """
    Finds H_perfect (or any reference 3D channel array) in the source file,
    and transposes H_est (which has shape (N, 132, 14)) to match its layout exactly.
    """
    import h5py
    with h5py.File(source_mat_path, 'r') as f:
        ref_shape = None
        for k in ['H_perfect', 'H_perfect_ori', 'H_li', 'H_prac']:
            if k in f and len(f[k].shape) == 3:
                ref_shape = f[k].shape
                break
        if ref_shape is None:
            for k in f.keys():
                if len(f[k].shape) == 3:
                    ref_shape = f[k].shape
                    break
                    
    if ref_shape is None:
        return np.transpose(H_est, (0, 2, 1))
        
    # Case 1: reference shape is (N, 14, 132) -> transpose to (0, 2, 1)
    if ref_shape[1] == 14 and ref_shape[2] == 132:
        return np.transpose(H_est, (0, 2, 1))
    # Case 2: reference shape is (N, 132, 14) -> keep as is
    elif ref_shape[1] == 132 and ref_shape[2] == 14:
        return H_est
    # Case 3: reference shape is (14, 132, N) -> transpose to (2, 1, 0)
    elif ref_shape[0] == 14 and ref_shape[1] == 132:
        return np.transpose(H_est, (2, 0, 1))
    # Case 4: reference shape is (132, 14, N) -> transpose to (1, 2, 0)
    elif ref_shape[0] == 132 and ref_shape[1] == 14:
        return np.transpose(H_est, (1, 2, 0))
        
    return np.transpose(H_est, (0, 2, 1))

def clip_sample_np(sample, row_min, row_max, col_min, col_max):
    """
    Clips the full grid's real/imag parts to the min/max values found in the pilot region.
    Matches clip_sample logic.
    """
    real = sample[:, :, 0]
    imag = sample[:, :, 1]
    
    roi_real = real[row_min:row_max, col_min:col_max]
    min_real = np.min(roi_real)
    max_real = np.max(roi_real)
    
    roi_imag = imag[row_min:row_max, col_min:col_max]
    min_imag = np.min(roi_imag)
    max_imag = np.max(roi_imag)
    
    real_clipped = np.clip(real, min_real, max_real)
    imag_clipped = np.clip(imag, min_imag, max_imag)
    
    return np.stack([real_clipped, imag_clipped], axis=-1)

def clip_batch_np(x, pilot_bounds):
    """
    Clips a batch of samples.
    """
    row_min, row_max, col_min, col_max = pilot_bounds
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = clip_sample_np(x[i], row_min, row_max, col_min, col_max)
    return out

def get_perfect_channel(f, input_key):
    """
    Finds the perfect reference channel dataset in the HDF5 file.
    """
    if 'H_perfect' in f:
        return h5_to_complex(f['H_perfect'][()])
    if 'H_perfect_ori' in f:
        return h5_to_complex(f['H_perfect_ori'][()])
    for k in f.keys():
        if k != input_key and len(f[k].shape) == 3:
            return h5_to_complex(f[k][()])
    raise KeyError("Perfect channel label not found in dataset file.")

def compute_mmse(H_pred, H_true):
    """Mean-squared error between predicted and true complex channels."""
    return float(np.mean(np.abs(H_pred - H_true) ** 2))

def compute_nmse(H_pred, H_true):
    """Normalised MSE (averaged over samples)."""
    num = np.mean(np.abs(H_pred - H_true) ** 2, axis=(1, 2))
    denom = np.mean(np.abs(H_true) ** 2, axis=(1, 2))
    return float(np.mean(num / (denom + 1e-30)))

def compute_nmse_db(H_pred, H_true):
    """NMSE in dB."""
    return 10.0 * np.log10(compute_nmse(H_pred, H_true) + 1e-30)

def compute_ssim_batch(H_pred, H_true):
    """
    SSIM computed on the magnitude of the complex channel images using SciPy
    (bypasses TensorFlow's NumPy 2.0 compatibility crash).
    """
    import scipy.ndimage
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
        
        # 2D Gaussian filter (sigma=1.5, size=11 via truncate=3.5)
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
        
        ssim_val = np.mean(num / denom)
        ssim_list.append(ssim_val)
        
    return float(np.mean(ssim_list))

def save_inferred_to_mat(dest_path, source_path, key, H_est, metrics_dict, num_samples=None):
    """
    Saves the entire contents of the original source_path MAT file combined
    with the inferred channel and metrics to dest_path using scipy.io.savemat.
    If num_samples is specified, slices all sample-dependent variables accordingly.
    """
    save_dict = {}
    
    # 1. Read all variables from the original HDF5 MAT file
    with h5py.File(source_path, 'r') as f:
        # Determine the original number of samples
        N_orig = None
        for k in ['H_perfect', 'H_perfect_ori', 'H_li', 'H_prac']:
            if k in f:
                N_orig = f[k].shape[0]
                break
        if N_orig is None:
            # Fallback scan
            for k in f.keys():
                if hasattr(f[k], 'shape') and len(f[k].shape) >= 1:
                    N_orig = f[k].shape[0]
                    break
        
        # Load and slice each variable
        for k in f.keys():
            if k.startswith('#') or k.startswith('__'):
                continue
                
            val = f[k][()]
            
            # Convert complex compound types to standard NumPy complex arrays
            val = h5_to_complex(val)
            
            # Slice sample-dependent variables
            if num_samples is not None and N_orig is not None:
                if hasattr(val, 'shape') and len(val.shape) >= 1 and val.shape[0] == N_orig:
                    val = val[:num_samples]
                    
            save_dict[k] = val

    # 2. Append new inference variables and metrics
    save_dict[key] = H_est
    save_dict['mmse'] = metrics_dict['mmse']
    save_dict['nmse'] = metrics_dict['nmse']
    save_dict['nmse_db'] = metrics_dict['nmse_db']
    save_dict['ssim'] = metrics_dict['ssim']
    
    # 3. Save to dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    scipy.io.savemat(dest_path, save_dict)
    print(f"  [SUCCESS] Saved combined MAT file to: {dest_path}")

def run_inference(model_dir=MODEL_DIR, dataset_dir=DATASET_DIR, num_samples=NUM_SAMPLES,
                  model_name=MODEL_NAME, clip_extrap=CLIP_EXTRAP, output_key=OUTPUT_KEY,
                  out_dir=OUT_DIR, model_type=MODEL_TYPE):
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' does not exist.")
        sys.exit(1)
        
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        sys.exit(1)

    # Determine extrapolation clipping mode
    if str(clip_extrap).lower() == "auto":
        is_clip_extrap = "clip" in os.path.basename(model_dir).lower()
        print(f"[Auto-detect] Clip extrapolation: {is_clip_extrap} (based on model folder name)")
    else:
        is_clip_extrap = str(clip_extrap).lower() == "true"
        print(f"[Manual override] Clip extrapolation: {is_clip_extrap}")

    print("\n" + "="*80)
    print("  ONNX Batch Inference for MATLAB Channels")
    print("="*80)
    print(f"Model Directory  : {model_dir}")
    print(f"Dataset Directory: {dataset_dir}")
    print(f"Output Directory : {out_dir if out_dir is not None else 'In-place (same as Dataset Directory)'}")
    print(f"Model Filename   : {model_name}")
    print(f"Clip Extrap Mode : {is_clip_extrap}")
    print(f"Output Key Mode  : {output_key}")
    print(f"Model Type Filter: {model_type}")
    print(f"Num Samples Limit: {num_samples if num_samples is not None else 'All'}")
    print("="*80 + "\n")

    results_summaries = {
        'LI': [],
        'LS': [],
        'PRAC': []
    }

    # List subfolders of the model directory
    model_subfolders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
    
    # Filter based on MODEL_TYPE parameter
    model_type_lower = str(model_type).lower()
    if model_type_lower in ["li", "ls", "prac"]:
        prefix_pattern = model_type_lower + "_"
        model_subfolders = [f for f in model_subfolders if f.lower().startswith(prefix_pattern)]
        print(f"[Filter] Applied model type filter '{model_type}'; matching folders: {model_subfolders}")
        
    # Sort subfolders to have a neat progress log
    model_subfolders = sorted(model_subfolders, key=lambda x: extract_snr(x) if extract_snr(x) is not None else 999)

    matched_count = 0

    for sub in model_subfolders:
        snr = extract_snr(sub)
        if snr is None:
            continue
            
        print(f"\nProcessing subfolder: {sub} (Detected SNR: {snr} dB)")
        
        # 1. Search for ONNX model
        model_paths_to_check = [
            os.path.join(model_dir, sub, "results", model_name),
            os.path.join(model_dir, sub, model_name)
        ]
        onnx_path = None
        for path in model_paths_to_check:
            if os.path.exists(path):
                onnx_path = path
                break
                
        if onnx_path is None:
            print(f"  [Warning] ONNX model '{model_name}' not found under {sub}. Skipping.")
            continue
            
        print(f"  Found ONNX model: {onnx_path}")
        
        # 2. Determine input key based on folder name prefix
        # Default maps LI -> H_li, LS -> H_ls, PRAC -> H_prac
        sub_lower = sub.lower()
        if "prac" in sub_lower:
            input_key = "H_prac"
        elif "ls" in sub_lower:
            input_key = "H_ls_pilots"
        else:
            input_key = "H_li"
            
        print(f"  Assumed input key: {input_key}")

        # 3. Find corresponding dataset folder
        dataset_subfolders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        target_dataset_sub = None
        for d in dataset_subfolders:
            d_snr = extract_snr(d)
            if d_snr == snr:
                target_dataset_sub = d
                break
                
        if target_dataset_sub is None:
            print(f"  [Warning] No matching SNR subfolder found in dataset directory for SNR {snr} dB. Skipping.")
            continue
            
        source_mat_path = os.path.join(dataset_dir, target_dataset_sub, "matlabNTN.mat")
        if not os.path.exists(source_mat_path):
            print(f"  [Warning] Dataset file matlabNTN.mat not found at '{source_mat_path}'. Skipping.")
            continue
            
        prefix = "LS" if "ls" in sub_lower else ("PRAC" if "prac" in sub_lower else "LI")
        dest_folder_name = f"{prefix}_{snr}dB"
        if out_dir is not None:
            dest_folder = os.path.join(out_dir, dest_folder_name)
            os.makedirs(dest_folder, exist_ok=True)
            dest_mat_path = os.path.join(dest_folder, "inferredChannel.mat")
        else:
            dest_folder = os.path.join(dataset_dir, target_dataset_sub)
            dest_mat_path = os.path.join(dest_folder, f"inferredChannel_{prefix}.mat")
            
        mat_path = source_mat_path
        print(f"  Matched target dataset: {mat_path}")
        print(f"  Destination file: {dest_mat_path}")

        # 4. Load dataset
        try:
            with h5py.File(mat_path, 'r') as f:
                # Resolve key name for LS if needed
                if "ls" in sub_lower:
                    ls_key = "H_ls_pilots"
                    if ls_key not in f:
                        for alt in ["H_ls_pilots", "H_ls_pilots_ori", "H_LS_comp", "H_LS_full"]:
                            if alt in f:
                                ls_key = alt
                                break
                    input_key = ls_key

                if input_key not in f:
                    print(f"    [Error] Key '{input_key}' not found in {mat_path}. Skipping SNR {snr}.")
                    continue
                
                H_perf_complex = get_perfect_channel(f, input_key)
                if H_perf_complex.ndim == 3:
                    if H_perf_complex.shape[1] == 14 and H_perf_complex.shape[2] == 132:
                        H_perf_complex = np.transpose(H_perf_complex, (0, 2, 1))

                # Check for LS vs LI/PRAC loading
                if "ls" in sub_lower:
                    # Sparse grid reconstruction for LS
                    if 'pilot_rows' not in f or 'pilot_cols' not in f:
                        print(f"    [Error] Sparse grid reconstruction keys (pilot_rows/pilot_cols) not found in {mat_path}. Skipping.")
                        continue
                    pilot_rows = np.squeeze(f['pilot_rows'][()]).astype(int) - 1
                    pilot_cols = np.squeeze(f['pilot_cols'][()]).astype(int) - 1
                    
                    H_pilots = h5_to_complex(f[input_key][()])
                    N_samples = H_perf_complex.shape[0]
                    if H_pilots.shape[0] != N_samples and H_pilots.shape[1] == N_samples:
                        H_pilots = H_pilots.T
                        
                    n_subc = H_perf_complex.shape[1]
                    n_symb = H_perf_complex.shape[2]
                    
                    H_in_complex = np.zeros((N_samples, n_subc, n_symb), dtype=np.complex128)
                    for i in range(N_samples):
                        H_in_complex[i, pilot_rows, pilot_cols] = H_pilots[i, :]
                        
                    orig_shape = H_in_complex.shape
                else:
                    H_in_dataset = f[input_key][()]
                    orig_shape = H_in_dataset.shape
                    H_in_complex = h5_to_complex(H_in_dataset)
                    
                    if H_in_complex.ndim == 3:
                        if H_in_complex.shape[1] == 14 and H_in_complex.shape[2] == 132:
                            H_in_complex = np.transpose(H_in_complex, (0, 2, 1))

                if num_samples is not None:
                    if num_samples <= 0:
                        print(f"    [Error] num-samples must be positive (got {num_samples}). Skipping.")
                        continue
                    if num_samples > H_in_complex.shape[0]:
                        print(f"    [Warning] num-samples ({num_samples}) is larger than dataset size ({H_in_complex.shape[0]}). Using all samples.")
                    else:
                        H_in_complex = H_in_complex[:num_samples]
                        H_perf_complex = H_perf_complex[:num_samples]
                        print(f"    Sliced dataset to first {num_samples} samples.")
                
                N_samples, n_subc, n_symb = H_in_complex.shape
                print(f"    Loaded input '{input_key}' of shape {H_in_complex.shape} and label H_perfect of shape {H_perf_complex.shape}")

                # Load pilot bounds for clipping
                pilot_rows = np.squeeze(f['pilot_rows'][()]).astype(int) - 1
                pilot_cols = np.squeeze(f['pilot_cols'][()]).astype(int) - 1
                row_min = int(np.min(pilot_rows))
                row_max = int(np.max(pilot_rows))
                col_min = int(np.min(pilot_cols))
                col_max = int(np.max(pilot_cols))
                pilot_bounds = (row_min, row_max + 1, col_min, col_max + 1)
        except Exception as e:
            print(f"    [Error] Failed to load dataset {mat_path}: {e}. Skipping.")
            continue

        # 5. Preprocess (Real/Imag stacking + Clipping + Normalization)
        start_time = time.perf_counter()
        
        # Convert complex (N, 132, 14) -> (N, 132, 14, 2)
        x = np.stack([H_in_complex.real, H_in_complex.imag], axis=-1).astype(np.float32)
        
        if is_clip_extrap:
            x = clip_batch_np(x, pilot_bounds)
            print(f"    Applied extrapolation clipping using pilot bounds: rows=[{row_min}, {row_max}], cols=[{col_min}, {col_max}]")

        x_scaled, x_min, x_max = minmax_scaler(x, lower_range=-1)

        # 6. Run ONNX Session Inference
        try:
            sess = ort.InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            input_shape = sess.get_inputs()[0].shape
            
            # Get expected input and output shapes
            input_shape = sess.get_inputs()[0].shape
            output_shape = sess.get_outputs()[0].shape
            
            print(f"    ONNX Model Shapes: Input={input_shape}, Output={output_shape}")

            # Determine input permutation mapping (1, 132, 14, 2) -> expected input shape
            input_perm = None
            if len(input_shape) == 4:
                perm = []
                for dim in input_shape:
                    if dim == 2:
                        perm.append(3)
                    elif dim == 132:
                        perm.append(1)
                    elif dim == 14:
                        perm.append(2)
                    else:
                        perm.append(0) # Batch dimension
                input_perm = tuple(perm)

            residuals = []
            for idx in range(N_samples):
                x_sample = x_scaled[idx:idx+1]  # Shape: (1, 132, 14, 2)
                
                # Transpose input to match model shape if permutation is found
                if input_perm is not None:
                    x_sample = np.transpose(x_sample, input_perm)
                
                # Run ONNX inference
                res_out = sess.run([output_name], {input_name: x_sample.astype(np.float32)})[0]
                
                # Dynamically align output back to shape (132, 14, 2)
                # Find dimensions of size 132, 14, and 2 in the output shape
                res_squeezed = np.squeeze(res_out)
                sq_shape = list(res_squeezed.shape)
                if 132 in sq_shape and 14 in sq_shape and 2 in sq_shape:
                    idx_132 = sq_shape.index(132)
                    idx_14 = sq_shape.index(14)
                    idx_2 = sq_shape.index(2)
                    res_aligned = np.transpose(res_squeezed, (idx_132, idx_14, idx_2))
                else:
                    # Fallback to standard NHWC shape
                    res_aligned = res_squeezed.reshape(132, 14, 2)
                
                residuals.append(res_aligned)
                
            residual = np.array(residuals)  # Shape: (N, 132, 14, 2)
        except Exception as e:
            print(f"    [Error] ONNX model execution failed: {e}. Skipping.")
            continue

        # 7. Postprocess (Denormalization + Complex reconstruction)
        x_corr = x_scaled + residual
        x_denormed = de_min_max(x_corr, x_min, x_max, lower_range=-1)
        H_est = x_denormed[..., 0] + 1j * x_denormed[..., 1]
        
        # Match original shape dynamically to ensure 100% same layout
        H_est_to_write = match_original_shape(H_est, source_mat_path)
        
        # End timing and calculate statistics
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_sample = total_time / N_samples

        # 8. Calculate average metrics
        mmse = compute_mmse(H_est, H_perf_complex)
        nmse = compute_nmse(H_est, H_perf_complex)
        nmse_db = compute_nmse_db(H_est, H_perf_complex)
        
        try:
            ssim = compute_ssim_batch(H_est, H_perf_complex)
        except Exception as e:
            print(f"    [Warning] Failed to compute SSIM: {e}")
            ssim = float('nan')

        print(f"    [Metrics] MMSE: {mmse:.6e} | NMSE: {nmse:.6f} | NMSE (dB): {nmse_db:.2f} dB | SSIM: {ssim:.6f}")

        # Determine out_key
        if output_key == "auto":
            if "prac" in sub_lower:
                out_key = "H_PRAC_infer"
            elif "ls" in sub_lower:
                out_key = "H_LS_infer"
            else:
                out_key = "H_LI_infer"
        else:
            out_key = output_key

        # Store for reference in summary
        results_summaries[prefix].append({
            'snr': snr,
            'mmse': mmse,
            'nmse': nmse,
            'nmse_db': nmse_db,
            'ssim': ssim,
            'out_key': out_key
        })

        # 9. Save inferred channel and metrics combined with original dataset variables
        metrics_dict = {
            'mmse': mmse,
            'nmse': nmse,
            'nmse_db': nmse_db,
            'ssim': ssim
        }
        save_inferred_to_mat(dest_mat_path, source_mat_path, out_key, H_est_to_write, metrics_dict, num_samples=num_samples)
        
        # 10. Save inference time report to .md note
        try:
            md_path = os.path.join(dest_folder, "inference_time_note.md")
            with open(md_path, 'w') as fh:
                fh.write(f"# Inference Time Report\n\n")
                fh.write(f"* **Model/Subfolder:** `{sub}`\n")
                fh.write(f"* **Total Samples in Batch:** {N_samples}\n")
                fh.write(f"* **Total Inference Time (including Pre + Postprocessing):** {total_time:.4f} seconds\n")
                fh.write(f"* **Average Inference Time per Sample:** {avg_time_per_sample * 1000:.4f} milliseconds ({avg_time_per_sample:.6f} seconds)\n")
            print(f"    [Report] Saved inference time note to: {md_path}")
        except Exception as report_err:
            print(f"    [Warning] Failed to write inference time note: {report_err}")

        matched_count += 1

    print("\n" + "="*80)
    print(f"  Inference batch run complete. Successfully saved {matched_count} datasets.")
    print("="*80 + "\n")

    # Write separate type-specific info files if out_dir is specified
    if out_dir is not None and matched_count > 0:
        # Helper to get notable path highlights starting from token
        def get_notable_path(full_path, start_token):
            normalized = os.path.normpath(full_path)
            match_idx = normalized.lower().find(start_token.lower())
            if match_idx != -1:
                return normalized[match_idx:]
            return os.path.basename(normalized)
            
        notable_model = get_notable_path(model_dir, "single_dataset")
        notable_dataset = get_notable_path(dataset_dir, "MATLAB")

        for prefix, summary_list in results_summaries.items():
            if not summary_list:
                continue
                
            info_filename = f"{prefix}_info.md"
            info_path = os.path.join(out_dir, info_filename)
            
            # Build metrics summary table
            table_rows = [
                "| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |",
                "|----------|------|------|-----------|------|"
            ]
            sorted_summary = sorted(summary_list, key=lambda x: x['snr'])
            for r in sorted_summary:
                table_rows.append(f"| {r['snr']:+d} | {r['mmse']:.6e} | {r['nmse']:.6f} | {r['nmse_db']:.2f} dB | {r['ssim']:.6f} |")
            metrics_table = "\n".join(table_rows)
            
            ref_out_key = sorted_summary[0]['out_key']
            
            md_content = f"""# {prefix} Inference Run Reference

- **Source Trained Model Folder**: {notable_model}
- **Target Dataset Folder**: {notable_dataset}

## Inference Performance Summary ({prefix})
{metrics_table}

## Inferred MAT File Field Reference
All variables are saved combined in **`inferredChannel.mat`** inside each target `{prefix}_xdB` subfolder.

### Belong to Inference Results
- `{ref_out_key}`: The complex estimated/inferred channel matrix (shape: `(N, 132, 14)`).
- `mmse`: Average Mean Squared Error compared to perfect label (scalar).
- `nmse`: Average Normalized Mean Squared Error (scalar).
- `nmse_db`: Average NMSE in dB (scalar).
- `ssim`: Average Structural Similarity Index (scalar).

### Belong to Original Dataset
- `H_li`: Original linear-interpolated input channel (shape: `(N, 132, 14)`).
- `H_ls_pilots`: Original sparse pilot values (shape: `(N, 88)`).
- `H_prac`: Original practical estimated channel (shape: `(N, 132, 14)`).
- `H_perfect` / `H_perfect_ori`: True channel labels (shape: `(N, 132, 14)`).
- `pilot_rows` / `pilot_cols` / `pilot_indices`: Grid positions of the pilot symbols.
- Sim geometry & propagation vectors: `r_ue_ECEF_all`, `ut_loc_ENU_all`, `slant_ranges`, `doppler_shifts_all`, `pl_dB_all`, `elevation_angles`, etc.
- Constant system variables: `bs_loc_ENU`, `r_sat_ECEF`, `v_sat_ECEF`, `v_sat_ENU`, `satelliteDopplerShift_bc`, etc.

## Inference Details
- **ONNX Model File**: {model_name}
- **Number of Samples**: {num_samples if num_samples is not None else 'All'}
- **Extrapolation Clipping**: {is_clip_extrap}
- **MATLAB Variable Key**: {ref_out_key}
"""
            try:
                with open(info_path, "w") as fh:
                    fh.write(md_content)
                print(f"  [SUCCESS] Created {info_filename} reference file in output directory: {info_path}")
            except Exception as e:
                print(f"  [Warning] Failed to write {info_filename}: {e}")

if __name__ == "__main__":
    run_inference()
