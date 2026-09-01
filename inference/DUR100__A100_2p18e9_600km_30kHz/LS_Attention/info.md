# Inference Run Reference

- **Source Trained Model Folder**: single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LS_Attention
- **Target Dataset Folder**: MATLAB\A100_2p18e9_600km_70deg_30kHz

## Inference Performance Summary
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 1.298527e-16 | 0.531221 | -2.75 dB | 0.278057 |
| -5 | 6.998334e-17 | 0.286641 | -5.43 dB | 0.347029 |
| +0 | 3.975076e-17 | 0.185486 | -7.32 dB | 0.412298 |
| +5 | 1.902107e-17 | 0.091801 | -10.37 dB | 0.590682 |
| +10 | 1.215638e-17 | 0.065834 | -11.82 dB | 0.662728 |
| +15 | 8.666375e-18 | 0.053410 | -12.72 dB | 0.728230 |

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
- **Standardization**: False
- **MATLAB Variable Key**: H_LS_infer
