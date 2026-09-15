# Inference Run Reference

- **Source Trained Model Folder**: single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_AxialAttention
- **Target Dataset Folder**: DUR100nsFix_pos2_2p18G_600km_70deg_r15km_20to30mps

## Inference Performance Summary
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 1.131488e-19 | 0.492376 | -3.08 dB | 0.719846 |
| -5 | 5.945273e-20 | 0.238899 | -6.22 dB | 0.791568 |
| +0 | 2.644192e-20 | 0.109713 | -9.60 dB | 0.858323 |
| +5 | 1.433388e-20 | 0.061040 | -12.14 dB | 0.901185 |
| +10 | 7.401501e-21 | 0.034120 | -14.67 dB | 0.940315 |

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
