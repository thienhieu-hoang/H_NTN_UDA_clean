# Inference Run Reference

- **Source Trained Model Folder**: single_dataset\pseudo_A100Perfect_DUR300LSAttention_minmax_13x5\LS_Attention_minmaxFDA_rmsReTrain
- **Target Dataset Folder**: DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps

## Inference Performance Summary
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 1.383722e-19 | 0.553290 | -2.57 dB | 0.953035 |
| -5 | 1.186843e-19 | 0.380601 | -4.20 dB | 0.956804 |
| +0 | 1.504824e-19 | 0.234982 | -6.29 dB | 0.967672 |
| +5 | 5.410340e-20 | 0.167597 | -7.76 dB | 0.975189 |
| +10 | 5.945843e-20 | 0.139096 | -8.57 dB | 0.979256 |
| +15 | 4.156063e-20 | 0.115861 | -9.36 dB | 0.980387 |

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
