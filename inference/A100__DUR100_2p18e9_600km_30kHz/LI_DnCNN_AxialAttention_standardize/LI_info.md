# LI Inference Run Reference

- **Source Trained Model Folder**: single_dataset\A100_2p18e9_600km_70deg_30kHz\LI_DnCNN_AxialAttention_standardize
- **Target Dataset Folder**: DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps

## Inference Performance Summary (LI)
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 7.734908e-17 | 298.677734 | 24.75 dB | 0.010177 |
| -5 | 1.665698e-17 | 64.438042 | 18.09 dB | 0.053785 |
| +0 | 5.564239e-18 | 21.916866 | 13.41 dB | 0.146767 |
| +5 | 1.350568e-18 | 5.480161 | 7.39 dB | 0.367855 |
| +10 | 3.622495e-19 | 1.452187 | 1.62 dB | 0.629676 |
| +15 | 1.121152e-19 | 0.398285 | -4.00 dB | 0.809530 |

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
