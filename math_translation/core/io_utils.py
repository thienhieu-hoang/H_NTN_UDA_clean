"""
I/O Utilities for Channel Datasets
Handles MATLAB .mat files (v7 and v7.3 HDF5), pilot coordinate extraction,
and channel tensor transformations.
"""

import os
import numpy as np
import scipy.io
import h5py


def h5_to_complex(dataset_value):
    """
    Converts h5py compound dataset (real, imag) or struct to standard complex numpy array.
    """
    if isinstance(dataset_value, np.ndarray) and dataset_value.dtype.names is not None:
        if 'real' in dataset_value.dtype.names and 'imag' in dataset_value.dtype.names:
            return dataset_value['real'] + 1j * dataset_value['imag']
    return dataset_value


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
        data_dict = {}
        with h5py.File(mat_path, 'r') as f:
            for k in f.keys():
                if not k.startswith('#') and not k.startswith('__'):
                    val = f[k][()]
                    data_dict[k] = h5_to_complex(val)
        return data_dict


def load_channel_mat(mat_path, input_type="ls"):
    """
    Loads channel dataset from MAT file or folder containing MAT file.
    Returns:
        H_perfect: (N, 132, 14) complex array
        H_pilots: (N, 88) complex array (or pilots extracted at pilot positions)
        pilot_rows: (88,) 0-based subcarrier indices
        pilot_cols: (88,) 0-based symbol indices
        mat_dict: full dictionary of variables
    """
    if os.path.isdir(mat_path):
        candidates = ["matlabNTN.mat", "channel_dur_randomizedUE.mat"]
        found = None
        for c in candidates:
            p = os.path.join(mat_path, c)
            if os.path.isfile(p):
                found = p
                break
        if found is None:
            mat_files = [f for f in os.listdir(mat_path) if f.endswith('.mat') and not f.startswith('inferredChannel')]
            if mat_files:
                found = os.path.join(mat_path, mat_files[0])
            else:
                raise FileNotFoundError(f"No MAT file found in directory: {mat_path}")
        mat_path = found

    try:
        mat = scipy.io.loadmat(mat_path)
        is_hdf5 = False
    except NotImplementedError:
        is_hdf5 = True

    if is_hdf5:
        with h5py.File(mat_path, 'r') as f:
            H_perfect = f['H_perfect'][()]
            H_perfect = h5_to_complex(H_perfect)
            # Layout adjustment: if (N, 14, 132) -> transpose to (N, 132, 14)
            if H_perfect.ndim == 3 and H_perfect.shape[1] == 14 and H_perfect.shape[2] == 132:
                H_perfect = np.transpose(H_perfect, (0, 2, 1))

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

            # Pilot coordinates (convert 1-based MATLAB to 0-based Python)
            pilot_rows = np.asarray(mat_dict.get('pilot_rows', np.array([])), dtype=int).squeeze()
            pilot_cols = np.asarray(mat_dict.get('pilot_cols', np.array([])), dtype=int).squeeze()
            if pilot_rows.size > 0 and pilot_rows.min() >= 1:
                pilot_rows = pilot_rows - 1
            if pilot_cols.size > 0 and pilot_cols.min() >= 1:
                pilot_cols = pilot_cols - 1

            input_key_map = {
                'prac': 'H_prac',
                'li': 'H_li',
                'li_ori': 'H_li_ori',
                'ls': 'H_ls_pilots',
                'ls_ori': 'H_ls_pilots_ori'
            }
            input_key = input_key_map.get(input_type, 'H_ls_pilots')
            if input_key not in mat_dict:
                for alt in ['H_ls_pilots', 'H_ls_pilots_ori', 'H_li', 'H_prac']:
                    if alt in mat_dict:
                        input_key = alt
                        break

            H_pilots = mat_dict.get(input_key, None)
            if H_pilots is not None and H_pilots.ndim == 3 and pilot_cols.size > 0:
                H_pilots = H_pilots[:, pilot_cols, pilot_rows]

    else:
        # Standard scipy.io.loadmat
        H_perfect = mat['H_perfect'].T
        mat_dict = mat

        pilot_rows = np.asarray(mat.get('pilot_rows', np.array([])), dtype=int).squeeze()
        pilot_cols = np.asarray(mat.get('pilot_cols', np.array([])), dtype=int).squeeze()
        if pilot_rows.size > 0 and pilot_rows.min() >= 1:
            pilot_rows = pilot_rows - 1
        if pilot_cols.size > 0 and pilot_cols.min() >= 1:
            pilot_cols = pilot_cols - 1

        input_key_map = {
            'prac': 'H_prac',
            'li': 'H_li',
            'li_ori': 'H_li_ori',
            'ls': 'H_ls_pilots',
            'ls_ori': 'H_ls_pilots_ori'
        }
        input_key = input_key_map.get(input_type, 'H_ls_pilots')
        if input_key not in mat or mat[input_key].size == 0:
            for alt in ['H_ls_pilots', 'H_ls_pilots_ori', 'H_li', 'H_prac']:
                if alt in mat and isinstance(mat[alt], np.ndarray) and mat[alt].size > 0:
                    input_key = alt
                    break

        H_pilots = mat[input_key].T
        if H_pilots.ndim == 3 and pilot_cols.size > 0:
            H_pilots = H_pilots[:, pilot_cols, pilot_rows]

    # If pilot coordinates are missing, provide default 5G NR DM-RS Type 2 indices
    if pilot_rows.size == 0 or pilot_cols.size == 0:
        pilot_rows, pilot_cols = get_default_dmrs_coords()

    # If H_pilots is missing, extract directly from H_perfect
    if H_pilots is None:
        H_pilots = H_perfect[:, pilot_rows, pilot_cols]

    return H_perfect, H_pilots, pilot_rows, pilot_cols, mat_dict


def get_default_dmrs_coords(n_subcarriers=132):
    """
    Generates default 5G NR DM-RS Type 2 pilot coordinates for 11 RBs (132 subcarriers)
    and 14 OFDM symbols. Type A pos 2 (col 2) + Add pos 1 (col 11) in 0-based indexing.
    Returns:
        pilot_rows: (88,) array of subcarrier indices
        pilot_cols: (88,) array of symbol indices
    """
    subcarriers = []
    for rb in range(11):
        base = rb * 12
        subcarriers.extend([base + 0, base + 1, base + 6, base + 7])
    subcarriers = np.array(subcarriers, dtype=int)  # 44 subcarriers

    pilot_rows = np.concatenate([subcarriers, subcarriers])
    pilot_cols = np.concatenate([np.full(44, 2, dtype=int), np.full(44, 11, dtype=int)])
    return pilot_rows, pilot_cols


def save_channel_mat(dest_path, orig_dict, save_vars_dict):
    """
    Saves new variables merged with original dictionary into a MAT file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    out_dict = {}
    for k, v in orig_dict.items():
        if not k.startswith('__') and not k.startswith('#'):
            out_dict[k] = v
    for k, v in save_vars_dict.items():
        out_dict[k] = v
    scipy.io.savemat(dest_path, out_dict)
