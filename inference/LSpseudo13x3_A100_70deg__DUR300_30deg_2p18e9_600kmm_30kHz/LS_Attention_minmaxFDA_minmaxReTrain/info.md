# Inference Run Reference

- **Source Trained Model Folder**: single_dataset\pseudo_A100Perfect_DUR300LSAttention_minmax_13x3\LS_Attention_minmaxFDA_minmaxReTrain
- **Target Dataset Folder**: DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps

## Inference Performance Summary
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 1.854240e-19 | 0.654592 | -1.84 dB | 0.929595 |
| -5 | 1.068082e-19 | 0.378950 | -4.21 dB | 0.947871 |
| +0 | 1.605969e-19 | 0.255166 | -5.93 dB | 0.961106 |
| +5 | 5.610676e-20 | 0.165833 | -7.80 dB | 0.971545 |
| +10 | 5.812930e-20 | 0.142588 | -8.46 dB | 0.976156 |
| +15 | 3.952232e-20 | 0.117030 | -9.32 dB | 0.978215 |

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

### Visual Comparison Plots Saved Per SNR Folder
- `comparison_real_parts.png` / `.pdf`: Real part heatmaps for 4 diverse sample pairs (Label vs Inferred).
- `comparison_magnitudes.png` / `.pdf`: Magnitude heatmaps for 4 diverse sample pairs (Label vs Inferred).
- `comparison_real_and_magnitude.png` / `.pdf`: Side-by-side 4x4 matrix combining real parts and magnitudes.

## Inference Details
- **ONNX Model File**: auto
- **Number of Samples**: All
- **Extrapolation Clipping**: False
- **Normalization Mode**: minmax
- **MATLAB Variable Key**: H_LS_infer
