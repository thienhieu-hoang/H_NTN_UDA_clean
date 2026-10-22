# Inference Run Reference

- **Source Trained Model Folder**: single_dataset\pseudo_A100Perfect_DUR300LSAttention_std_13x3\LS_Attention_stdFDA_rmsReTrain
- **Target Dataset Folder**: DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps

## Inference Performance Summary
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 1.274560e-19 | 0.533246 | -2.73 dB | 0.951661 |
| -5 | 1.099307e-19 | 0.354750 | -4.50 dB | 0.957385 |
| +0 | 1.389232e-19 | 0.227256 | -6.43 dB | 0.966414 |
| +5 | 5.251944e-20 | 0.164839 | -7.83 dB | 0.974812 |
| +10 | 5.928254e-20 | 0.132418 | -8.78 dB | 0.979583 |
| +15 | 4.701852e-20 | 0.125404 | -9.02 dB | 0.978677 |

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
