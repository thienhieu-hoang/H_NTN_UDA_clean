# Inference Run Reference

- **Source Trained Model Folder**: single_dataset\pseudo_A100Perfect_DUR300LSAttention_minmax_13x3\LS_Attention_minmaxFDA_rmsReTrain
- **Target Dataset Folder**: DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps

## Inference Performance Summary
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 1.389469e-19 | 0.523094 | -2.81 dB | 0.951707 |
| -5 | 1.008115e-19 | 0.345609 | -4.61 dB | 0.958511 |
| +0 | 1.556596e-19 | 0.232249 | -6.34 dB | 0.967477 |
| +5 | 5.886385e-20 | 0.178093 | -7.49 dB | 0.975399 |
| +10 | 5.995455e-20 | 0.144087 | -8.41 dB | 0.979144 |
| +15 | 3.735443e-20 | 0.112110 | -9.50 dB | 0.979788 |

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
- **Normalization Mode**: rms
- **MATLAB Variable Key**: H_LS_infer
