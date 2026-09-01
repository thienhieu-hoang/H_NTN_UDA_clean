import os
import sys
import re
import shutil
import time
import numpy as np
import scipy.io
import h5py
import onnxruntime as ort
import argparse

# ============================================================================
# CONFIGURATION CONSTANTS
# Edit these paths and parameters directly to run from your IDE
# ============================================================================
MODEL_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_LS_Attention_standardize"
DATASET_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\MATLAB\sampleWiseDoppler_wGeometry_A100_2p18e9_600km_70deg_30kHz"
OUT_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\inference\DUR100__A100_2p18e9_600km_30kHz_LSSequence_standardize"
MODEL_NAME = "auto"          # "auto" adaptively detects best_net.onnx, best_model.onnx, final_model.onnx, final_net.onnx, etc.
CLIP_EXTRAP = "auto"       # "auto", "true", or "false" (auto detects true if "clip" or "LI" is in model_dir or subdirs)
STANDARDIZE = "auto"       # "auto" (detects if "standardize" is in model_dir name), True, or False      ## False hope = using min-max Scaler
OUTPUT_KEY = "auto"        # "auto" saves variable as "H_infer", otherwise custom string
NUM_SAMPLES = 512           # Integer (e.g. 512, 64) or None to process all samples in the dataset
# ============================================================================

def find_onnx_model(sub_dir, preferred_name="auto"):
    """
    Adaptively finds an ONNX model in sub_dir or sub_dir/results.
    Checks preferred_name (if not 'auto'), then best_net.onnx, best_model.onnx,
    final_model.onnx, final_net.onnx, and falls back to any *.onnx file found.
    """
    candidates = []
    if preferred_name and str(preferred_name).lower() not in ["auto", "none", ""]:
        candidates.append(preferred_name)
    
    default_candidates = [
        "best_net.onnx",
        "best_model.onnx",
        "final_model.onnx",
        "final_net.onnx"
    ]
    for c in default_candidates:
        if c not in candidates:
            candidates.append(c)
            
    search_dirs = [
        os.path.join(sub_dir, "results"),
        sub_dir
    ]
    
    # 1. Search for named candidates in priority order
    for c in candidates:
        for s_dir in search_dirs:
            p = os.path.join(s_dir, c)
            if os.path.isfile(p):
                return p
                
    # 2. Fallback: Search for any .onnx file in the directories
    for s_dir in search_dirs:
        if os.path.isdir(s_dir):
            onnx_files = [f for f in os.listdir(s_dir) if f.lower().endswith('.onnx')]
            if onnx_files:
                # Prioritize 'best' -> 'final' -> alphabetical
                onnx_files.sort(key=lambda x: (
                    0 if "best" in x.lower() else (1 if "final" in x.lower() else 2),
                    x.lower()
                ))
                return os.path.join(s_dir, onnx_files[0])
                
    return None

def extract_snr(folder_name):
    """
    Extracts the integer SNR value from a folder name.
    Matches cases like LS_-10, LS_-5, LS_0, LS_5, LS_10, LS_15, or SNR_-10dB, etc.
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
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1, 2)
    x_min = np.min(x_reshaped, axis=1)  # (N, 2)
    x_max = np.max(x_reshaped, axis=1)  # (N, 2)
    scale = np.clip(x_max - x_min, 1e-30, None)  # (N, 2)
    
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

def standardize_scaler(x):
    """
    Standardize numpy array sample-wise along sequence dimensions.
    x has shape (N, L, 2)
    """
    N = x.shape[0]
    # Calculate mean and std sample-wise over the L elements
    x_mean = np.mean(x, axis=1)  # (N, 2)
    x_std = np.std(x, axis=1)    # (N, 2)
    x_std = np.clip(x_std, 1e-30, None)  # (N, 2)
    
    reshape_dims = [N] + [1] * (x.ndim - 2) + [2]
    x_mean_b = x_mean.reshape(reshape_dims)
    scale_b = x_std.reshape(reshape_dims)
    
    x_scaled = (x - x_mean_b) / scale_b
    return x_scaled, x_mean, x_std

def de_standardize(x_normd, x_mean, x_std):
    """
    Perform inverse sample-wise standardization for x_normd.
    x_normd has shape (N, H, W, 2)
    """
    N = x_normd.shape[0]
    reshape_dims = [N] + [1] * (x_normd.ndim - 2) + [2]
    x_mean_b = x_mean.reshape(reshape_dims)
    x_std_b = x_std.reshape(reshape_dims)
    return x_normd * x_std_b + x_mean_b


def load_source_mat_file(mat_path):
    """
    Loads a .mat file supporting both MATLAB v7 (scipy.io.loadmat)
    and MATLAB v7.3 HDF5 (h5py.File), returning a dictionary of {var_name: numpy_array}.
    """
    try:
        raw_dict = scipy.io.loadmat(mat_path)
        data_dict = {}
        for k, v in raw_dict.items():
            if not k.startswith('__'):
                data_dict[k] = v
        return data_dict
    except NotImplementedError:
        # File is MATLAB v7.3 (HDF5 format)
        data_dict = {}
        with h5py.File(mat_path, 'r') as f:
            for k in f.keys():
                if not k.startswith('#') and not k.startswith('__'):
                    val = f[k][()]
                    data_dict[k] = h5_to_complex(val)
        return data_dict

def match_original_shape(H_est, source_data_or_path):
    """
    Finds H_perfect (or any reference 3D channel array) in the dataset,
    and transposes H_est (which has shape (N, 132, 14)) to match its layout exactly.
    """
    if isinstance(source_data_or_path, dict):
        data_dict = source_data_or_path
    else:
        data_dict = load_source_mat_file(source_data_or_path)

    ref_shape = None
    for k in ['H_perfect', 'H_perfect_ori', 'H_li', 'H_prac']:
        if k in data_dict and hasattr(data_dict[k], 'shape') and len(data_dict[k].shape) == 3:
            ref_shape = data_dict[k].shape
            break
    if ref_shape is None:
        for k in data_dict.keys():
            if hasattr(data_dict[k], 'shape') and len(data_dict[k].shape) == 3:
                ref_shape = data_dict[k].shape
                break
                
    if ref_shape is None:
        return H_est
        
    # Case 1: reference shape is (N, 14, 132) -> transpose to (0, 2, 1)
    if ref_shape[1] == 14 and ref_shape[2] == 132:
        return np.transpose(H_est, (0, 2, 1))
    # Case 2: reference shape is (N, 132, 14) -> keep as is
    elif ref_shape[1] == 132 and ref_shape[2] == 14:
        return H_est
    # Case 3: reference shape is (14, 132, N) -> transpose to (2, 1, 0)
    elif ref_shape[0] == 14 and ref_shape[1] == 132:
        return np.transpose(H_est, (2, 1, 0))
    # Case 4: reference shape is (132, 14, N) -> transpose to (1, 2, 0)
    elif ref_shape[0] == 132 and ref_shape[1] == 14:
        return np.transpose(H_est, (1, 2, 0))
        
    return H_est

def load_mat_data(mat_path: str, input_type: str):
    try:
        mat = scipy.io.loadmat(mat_path)
        is_hdf5 = False
    except NotImplementedError:
        import h5py
        is_hdf5 = True

    if is_hdf5:
        def h5_to_complex(val):
            if isinstance(val, np.ndarray) and val.dtype.names is not None:
                if 'real' in val.dtype.names and 'imag' in val.dtype.names:
                    return val['real'] + 1j * val['imag']
            return val

        with h5py.File(mat_path, 'r') as f:
            H_perfect = f['H_perfect'][()]
            H_perfect = h5_to_complex(H_perfect)
            if H_perfect.ndim == 3:
                if H_perfect.shape[1] == 14 and H_perfect.shape[2] == 132:
                    H_perfect = np.transpose(H_perfect, (0, 2, 1))

            # Reconstruct dictionary of other variables
            mat_dict = {}
            for k in f.keys():
                if k.startswith('#'):
                    continue
                mat_dict[k] = h5_to_complex(f[k][()])
                if k in ['pilot_rows', 'pilot_cols']:
                    mat_dict[k] = np.squeeze(mat_dict[k])
                elif isinstance(mat_dict[k], np.ndarray) and mat_dict[k].ndim == 3:
                    if mat_dict[k].shape[1] == 14 and mat_dict[k].shape[2] == 132:
                        mat_dict[k] = np.transpose(mat_dict[k], (0, 2, 1))

            input_key_map = {
                'prac': 'H_prac',
                'li': 'H_li',
                'li_ori': 'H_li_ori',
                'ls': 'H_ls_pilots',
                'ls_ori': 'H_ls_pilots_ori'
            }
            input_key = input_key_map.get(input_type, 'H_ls_pilots')
            if input_key not in mat_dict:
                for alt in [input_key, 'H_ls_pilots', 'H_ls_pilots_ori', 'H_li', 'H_li_ori', 'H_prac']:
                    if alt in mat_dict:
                        input_key = alt
                        break

            H_input_pilots = mat_dict[input_key]
            pilot_cols = mat_dict['pilot_cols'].squeeze() - 1
            pilot_rows = mat_dict['pilot_rows'].squeeze() - 1

            if H_input_pilots.ndim == 3:
                H_input_pilots = H_input_pilots[:, pilot_cols, pilot_rows]

            H_li_benchmark_grid = mat_dict.get('H_li', mat_dict.get('H_li_ori', mat_dict.get('H_perfect')))
            if H_li_benchmark_grid.ndim == 2:
                H_li_benchmark_grid = H_perfect

    else:
        H_perfect = mat['H_perfect'].T
        
        input_key_map = {
            'prac': 'H_prac',
            'li': 'H_li',
            'li_ori': 'H_li_ori',
            'ls': 'H_ls_pilots',
            'ls_ori': 'H_ls_pilots_ori'
        }
        input_key = input_key_map.get(input_type, 'H_ls_pilots')
        
        if input_key not in mat or mat[input_key].size == 0:
            for alt in [input_key, 'H_ls_pilots', 'H_ls_pilots_ori', 'H_li', 'H_li_ori', 'H_prac']:
                if alt in mat and isinstance(mat[alt], np.ndarray) and mat[alt].size > 0:
                    input_key = alt
                    break
                    
        H_input_pilots = mat[input_key].T
        pilot_cols = mat['pilot_cols'].squeeze() - 1
        pilot_rows = mat['pilot_rows'].squeeze() - 1
        
        if H_input_pilots.ndim == 3:
            H_input_pilots = H_input_pilots[:, pilot_cols, pilot_rows]
            
        H_li_benchmark_grid = mat.get('H_li', mat.get('H_li_ori', mat['H_perfect'])).T
        if H_li_benchmark_grid.ndim == 2:
            H_li_benchmark_grid = H_perfect
            
        mat_dict = mat

    return H_perfect, H_input_pilots, H_li_benchmark_grid, mat_dict

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

def save_inferred_to_mat(dest_path, source_path_or_dict, key, H_est, metrics_dict, num_samples=None):
    """
    Saves the entire contents of the original source_path MAT file combined
    with the inferred channel and metrics to dest_path using scipy.io.savemat.
    If num_samples is specified, slices all sample-dependent variables accordingly.
    """
    save_dict = {}
    
    if isinstance(source_path_or_dict, dict):
        orig_dict = source_path_or_dict
    else:
        orig_dict = load_source_mat_file(source_path_or_dict)

    # Determine the original number of samples
    N_orig = None
    for k in ['H_perfect', 'H_perfect_ori', 'H_li', 'H_prac']:
        if k in orig_dict and hasattr(orig_dict[k], 'shape') and len(orig_dict[k].shape) >= 1:
            N_orig = orig_dict[k].shape[0]
            break
    if N_orig is None:
        for k, v in orig_dict.items():
            if hasattr(v, 'shape') and len(v.shape) >= 1:
                N_orig = v.shape[0]
                break
    
    # Load and slice each variable
    for k, val in orig_dict.items():
        if k.startswith('#') or k.startswith('__'):
            continue
            
        # Slice sample-dependent variables
        if num_samples is not None and N_orig is not None:
            if hasattr(val, 'shape') and len(val.shape) >= 1:
                if val.shape[0] == N_orig:
                    val = val[:num_samples]
                elif len(val.shape) == 3 and val.shape[2] == N_orig:
                    val = val[:, :, :num_samples]
                elif len(val.shape) == 2 and val.shape[1] == N_orig:
                    val = val[:, :num_samples]
                
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
                  out_dir=OUT_DIR, standardize=STANDARDIZE):
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' does not exist.")
        sys.exit(1)
        
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        sys.exit(1)

    # Determine extrapolation clipping mode
    if str(clip_extrap).lower() == "auto":
        is_clip_extrap = "clip" in model_dir.lower() or "li" in model_dir.lower()
        if not is_clip_extrap:
            try:
                subdirs = [d.lower() for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
                is_clip_extrap = any("li" in d for d in subdirs)
            except Exception:
                pass
        print(f"[Auto-detect] Clip extrapolation: {is_clip_extrap} (based on folder names/structure)")
    else:
        is_clip_extrap = str(clip_extrap).lower() == "true"
        print(f"[Manual override] Clip extrapolation: {is_clip_extrap}")

    # Determine standardization scaling mode
    if str(standardize).lower() == "auto":
        is_standardize = "standardize" in os.path.basename(model_dir).lower()
        print(f"[Auto-detect] Standardization: {is_standardize} (based on model folder name)")
    else:
        is_standardize = str(standardize).lower() == "true"
        print(f"[Manual override] Standardization: {is_standardize}")

    print("\n" + "="*80)
    print("  ONNX Batch Inference for MATLAB Channels")
    print("="*80)
    print(f"Model Directory  : {model_dir}")
    print(f"Dataset Directory: {dataset_dir}")
    print(f"Output Directory : {out_dir if out_dir is not None else 'In-place (same as Dataset Directory)'}")
    print(f"Model Filename   : {model_name}")
    print(f"Clip Extrap Mode : {is_clip_extrap}")
    print(f"Standardize Mode : {is_standardize}")
    print(f"Output Key Mode  : {output_key}")
    print(f"Num Samples Limit: {num_samples if num_samples is not None else 'All'}")
    print("="*80 + "\n")

    results_summary = []

    # List subfolders of the model directory
    model_subfolders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
    
    # Sort subfolders to have a neat progress log
    model_subfolders = sorted(model_subfolders, key=lambda x: extract_snr(x) if extract_snr(x) is not None else 999)

    matched_count = 0

    for sub in model_subfolders:
        snr = extract_snr(sub)
        if snr is None:
            continue
            
        print(f"\nProcessing subfolder: {sub} (Detected SNR: {snr} dB)")
        
        # 1. Search for ONNX model adaptively
        onnx_path = find_onnx_model(os.path.join(model_dir, sub), preferred_name=model_name)
        if onnx_path is None:
            print(f"  [Warning] ONNX model not found under {sub} (checked: best_net, best_model, final_model, final_net, etc.). Skipping.")
            continue
            
        detected_model_name = os.path.basename(onnx_path)
        print(f"  Found ONNX model: {onnx_path} (Adaptive: '{detected_model_name}')")
        
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
            target_dir_path = os.path.join(dataset_dir, target_dataset_sub)
            if os.path.exists(target_dir_path) and os.path.isdir(target_dir_path):
                mat_files = [f for f in os.listdir(target_dir_path) if f.endswith('.mat') and not f.startswith('inferredChannel')]
                if mat_files:
                    source_mat_path = os.path.join(target_dir_path, mat_files[0])
                    print(f"  [Auto-detect] matlabNTN.mat not found. Found other MAT file: '{source_mat_path}'")
                else:
                    print(f"  [Warning] No .mat file found under '{target_dir_path}'. Skipping.")
                    continue
            else:
                print(f"  [Warning] Dataset directory '{target_dir_path}' does not exist. Skipping.")
                continue
            
        prefix = "LS" if "ls" in sub_lower else ("PRAC" if "prac" in sub_lower else "LI")
        dest_folder_name = f"{prefix}_{snr}dB"
        if out_dir is not None:
            dest_folder = os.path.join(out_dir, dest_folder_name)
            os.makedirs(dest_folder, exist_ok=True)
            dest_mat_path = os.path.join(dest_folder, "inferredChannel.mat")
        else:
            dest_folder = os.path.join(dataset_dir, target_dataset_sub)
            dest_mat_path = os.path.join(dest_folder, "inferredChannel.mat")
            
        mat_path = source_mat_path
        print(f"  Matched target dataset: {mat_path}")
        print(f"  Destination file: {dest_mat_path}")

        # 4. Load dataset using updated load_mat_data function
        try:
            H_perf_complex, H_in_pilots, _, mat_dict = load_mat_data(mat_path, input_type="ls")
            
            # Slice dataset if num_samples is specified
            if num_samples is not None:
                H_in_pilots = H_in_pilots[:num_samples]
                H_perf_complex = H_perf_complex[:num_samples]
                
            N_samples = H_in_pilots.shape[0]
            print(f"    Loaded sequence input of shape {H_in_pilots.shape} and label H_perfect of shape {H_perf_complex.shape}")
        except Exception as e:
            print(f"    [Error] Failed to load dataset {mat_path}: {e}. Skipping.")
            continue

        # 5. Preprocess (Real/Imag stacking + Normalization)
        start_time = time.perf_counter()
        
        # Convert complex (N, P) -> (N, P, 2)
        x = np.stack([H_in_pilots.real, H_in_pilots.imag], axis=-1).astype(np.float32)
        
        # Scaling sample-wise over the pilot elements
        if is_standardize:
            x_scaled, x_mean, x_std = standardize_scaler(x)
        else:
            x_scaled, x_min, x_max = minmax_scaler(x, lower_range=-1)

        # 6. Run ONNX Session Inference
        try:
            sess = ort.InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            input_shape = sess.get_inputs()[0].shape
            
            print(f"    ONNX Input: Name='{input_name}', Shape={input_shape}")

            y_pred_scaled_list = []
            for idx in range(N_samples):
                x_sample = x_scaled[idx:idx+1]
                
                # Inference (expects (1, P, 2))
                y_out = sess.run([output_name], {input_name: x_sample.astype(np.float32)})[0]
                
                y_pred_scaled_list.append(y_out[0])
                
            y_pred_scaled = np.array(y_pred_scaled_list)  # (N, 132, 14, 2)
        except Exception as e:
            print(f"    [Error] ONNX model execution failed: {e}. Skipping.")
            continue

        # 7. Postprocess (Denormalization + Complex reconstruction)
        if is_standardize:
            x_denormed = de_standardize(y_pred_scaled, x_mean, x_std)
        else:
            x_denormed = de_min_max(y_pred_scaled, x_min, x_max, lower_range=-1)
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

        # Store for reference in info.md summary
        results_summary.append({
            'snr': snr,
            'mmse': mmse,
            'nmse': nmse,
            'nmse_db': nmse_db,
            'ssim': ssim
        })

        # 9. Save inferred channel and metrics combined with original dataset variables
        if output_key == "auto":
            if "prac" in sub_lower:
                out_key = "H_PRAC_infer"
            elif "ls" in sub_lower:
                out_key = "H_LS_infer"
            else:
                out_key = "H_LI_infer"
        else:
            out_key = output_key
        
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

    # Write info.md inside out_dir if specified
    if out_dir is not None and matched_count > 0:
        info_path = os.path.join(out_dir, "info.md")
        
        # Helper to get notable path highlights starting from token
        def get_notable_path(full_path, start_token):
            normalized = os.path.normpath(full_path)
            match_idx = normalized.lower().find(start_token.lower())
            if match_idx != -1:
                return normalized[match_idx:]
            return os.path.basename(normalized)
            
        notable_model = get_notable_path(model_dir, "single_dataset")
        notable_dataset = get_notable_path(dataset_dir, "MATLAB")
        
        # Build metrics summary table
        table_rows = [
            "| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |",
            "|----------|------|------|-----------|------|"
        ]
        sorted_summary = sorted(results_summary, key=lambda x: x['snr'])
        for r in sorted_summary:
            table_rows.append(f"| {r['snr']:+d} | {r['mmse']:.6e} | {r['nmse']:.6f} | {r['nmse_db']:.2f} dB | {r['ssim']:.6f} |")
        metrics_table = "\n".join(table_rows)
        
        md_content = f"""# Inference Run Reference

- **Source Trained Model Folder**: {notable_model}
- **Target Dataset Folder**: {notable_dataset}

## Inference Performance Summary
{metrics_table}

## Inferred MAT File Field Reference
All variables are saved combined in **`inferredChannel.mat`** inside each target SNR subfolder.

### Belong to Inference Results
- `{out_key}`: The complex estimated/inferred channel matrix (shape: `(N, 132, 14)`).
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
- **Standardization**: {is_standardize}
- **MATLAB Variable Key**: {out_key}
"""
        try:
            with open(info_path, "w") as fh:
                fh.write(md_content)
            print(f"  [SUCCESS] Created info.md reference file in output directory: {info_path}")
        except Exception as e:
            print(f"  [Warning] Failed to write info.md: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Batch Inference for LS Sequence Channel Estimation Models.")
    parser.add_argument('--model-dir', type=str, default=MODEL_DIR, help="Model directory containing ONNX file")
    parser.add_argument('--dataset-dir', type=str, default=DATASET_DIR, help="Dataset directory containing target MAT files")
    parser.add_argument('--out-dir', type=str, default=OUT_DIR, help="Output folder (None for in-place)")
    parser.add_argument('--num-samples', type=str, default=str(NUM_SAMPLES), help="Limit number of samples (None for all)")
    parser.add_argument('--model-type', type=str, default="auto", help="Filter folder types (for compatibility)")
    parser.add_argument('--clip-extrap', type=str, default=CLIP_EXTRAP, help="Extrapolation clipping (auto, true, false)")
    parser.add_argument('--standardize', type=str, default=STANDARDIZE, help="Standardization mode (auto, true, false)")
    parser.add_argument('--model-name', type=str, default=MODEL_NAME, help="ONNX model filename or 'auto'")
    parser.add_argument('--output-key', type=str, default=OUTPUT_KEY, help="MATLAB variable output key")
    
    args = parser.parse_args()
    
    # Resolve the num_samples None vs Int parsing
    nsamp = args.num_samples
    if nsamp.lower() in ['none', 'null', '']:
        nsamp = None
    else:
        try:
            nsamp = int(nsamp)
        except ValueError:
            nsamp = None
        
    run_inference(
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
        num_samples=nsamp,
        model_name=args.model_name,
        clip_extrap=args.clip_extrap,
        standardize=args.standardize,
        output_key=args.output_key,
        out_dir=args.out_dir if args.out_dir not in ['None', 'none', ''] else None
    )
