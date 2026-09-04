# LI Inference Run Reference

- **Source Trained Model Folder**: single_dataset\A100_2p18e9_600km_70deg_30kHz\LI_DnCNN_CrossAttention_standardize
- **Target Dataset Folder**: DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps

## Inference Performance Summary (LI)
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 5.923223e-17 | 229.756378 | 23.61 dB | 0.011983 |
| -5 | 1.745249e-17 | 67.470856 | 18.29 dB | 0.050499 |
| +0 | 5.978845e-18 | 23.816467 | 13.77 dB | 0.142633 |
| +5 | 1.722528e-18 | 6.779798 | 8.31 dB | 0.356617 |
| +10 | 4.076055e-19 | 1.616058 | 2.08 dB | 0.615173 |
| +15 | 1.188106e-19 | 0.424442 | -3.72 dB | 0.811363 |

## Inferred MAT File Field Reference
All variables are saved combined in **`inferredChannel.mat`** inside each target `LI_xdB` subfolder.

### Belong to Inference Results
- `H_LI_infer`: The complex estimated/inferred channel matrix (shape: `(N, 132, 14)`).
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
- **Number of Samples**: All
- **Extrapolation Clipping**: False
- **MATLAB Variable Key**: H_LI_infer
