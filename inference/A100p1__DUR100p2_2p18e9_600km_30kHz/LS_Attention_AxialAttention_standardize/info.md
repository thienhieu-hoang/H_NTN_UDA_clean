# Inference Run Reference

- **Source Trained Model Folder**: single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_AxialAttention_standardize
- **Target Dataset Folder**: DUR100nsFix_pos2_2p18G_600km_70deg_r15km_20to30mps

## Inference Performance Summary
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 8.730869e-20 | 0.373735 | -4.27 dB | 0.740940 |
| -5 | 5.161584e-20 | 0.209611 | -6.79 dB | 0.801260 |
| +0 | 2.077870e-20 | 0.087871 | -10.56 dB | 0.871300 |
| +5 | 1.024553e-20 | 0.045709 | -13.40 dB | 0.919554 |
| +10 | 6.214303e-21 | 0.029251 | -15.34 dB | 0.945590 |

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
