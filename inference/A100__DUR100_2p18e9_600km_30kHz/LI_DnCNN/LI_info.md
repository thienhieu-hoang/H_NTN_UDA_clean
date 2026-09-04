# LI Inference Run Reference

- **Source Trained Model Folder**: single_dataset\A100_2p18e9_600km_70deg_30kHz\LI_DnCNN
- **Target Dataset Folder**: DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps

## Inference Performance Summary (LI)
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 5.491408e-18 | 21.023167 | 13.23 dB | 0.172780 |
| -5 | 1.309765e-18 | 5.115123 | 7.09 dB | 0.397888 |
| +0 | 3.958108e-19 | 1.568689 | 1.96 dB | 0.589942 |
| +5 | 1.228352e-19 | 0.489330 | -3.10 dB | 0.759808 |
| +10 | 4.034026e-20 | 0.160726 | -7.94 dB | 0.881625 |
| +15 | 1.200658e-20 | 0.042771 | -13.69 dB | 0.948864 |

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
