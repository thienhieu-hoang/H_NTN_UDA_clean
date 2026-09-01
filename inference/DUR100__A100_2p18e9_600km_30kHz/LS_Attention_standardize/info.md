# Inference Run Reference

- **Source Trained Model Folder**: single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LS_Attention_standardize
- **Target Dataset Folder**: MATLAB\A100_2p18e9_600km_70deg_30kHz

## Inference Performance Summary
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 9.977849e-17 | 0.413792 | -3.83 dB | 0.366090 |
| -5 | 5.650533e-17 | 0.233551 | -6.32 dB | 0.432578 |
| +0 | 2.986296e-17 | 0.136789 | -8.64 dB | 0.531531 |
| +5 | 1.774836e-17 | 0.085686 | -10.67 dB | 0.611461 |
| +10 | 1.034906e-17 | 0.054579 | -12.63 dB | 0.696273 |
| +15 | 6.306368e-18 | 0.037535 | -14.26 dB | 0.750455 |

## Inferred MAT File Field Reference
All variables are saved combined in **`inferredChannel.mat`** inside each target SNR subfolder.

### Belong to Inference Results
- `H_LS_infer`: The complex estimated/inferred channel matrix (shape: `(N, 132, 14)`).
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
- **ONNX Model File**: auto
- **Number of Samples**: All
- **Extrapolation Clipping**: False
- **Standardization**: True
- **MATLAB Variable Key**: H_LS_infer
