# LS Inference Run Reference

- **Source Trained Model Folder**: single_dataset\DnCNN_Attention_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps
- **Target Dataset Folder**: MATLAB\sampleWiseDoppler_wGeometry_A100_2p18e9_600km_70deg_30kHz

## Inference Performance Summary (LS)
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 2.001345e-16 | 0.777737 | -1.09 dB | 0.081790 |
| -5 | 1.418830e-16 | 0.502113 | -2.99 dB | 0.096629 |
| +0 | 1.248046e-16 | 0.424242 | -3.72 dB | 0.147406 |
| +5 | 8.921563e-17 | 0.281740 | -5.50 dB | 0.226589 |
| +10 | 1.317091e-16 | 0.428061 | -3.68 dB | 0.252396 |
| +15 | 9.178349e-17 | 0.316897 | -4.99 dB | 0.268884 |

## Inferred MAT File Field Reference
All variables are saved combined in **`inferredChannel.mat`** inside each target `LS_xdB` subfolder.

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
- **ONNX Model File**: best_net.onnx
- **Number of Samples**: 512
- **Extrapolation Clipping**: False
- **MATLAB Variable Key**: H_LS_infer
