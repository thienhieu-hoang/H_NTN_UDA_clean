# Inference Run Reference

- **Source Trained Model Folder**: single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_LS_Attention_standardize
- **Target Dataset Folder**: MATLAB\sampleWiseDoppler_wGeometry_A100_2p18e9_600km_70deg_30kHz

## Inference Performance Summary
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 4.909129e-17 | 0.235002 | -6.29 dB | 0.481999 |
| -5 | 3.392949e-17 | 0.161641 | -7.91 dB | 0.547264 |
| +0 | 2.278557e-17 | 0.117955 | -9.28 dB | 0.609672 |
| +5 | 1.529975e-17 | 0.077402 | -11.11 dB | 0.661556 |
| +10 | 8.988440e-18 | 0.046582 | -13.32 dB | 0.732387 |
| +15 | 5.780288e-18 | 0.035869 | -14.45 dB | 0.770690 |

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
- **ONNX Model File**: best_model.onnx
- **Number of Samples**: 512
- **Extrapolation Clipping**: False
- **Standardization**: True
- **MATLAB Variable Key**: H_LS_infer
