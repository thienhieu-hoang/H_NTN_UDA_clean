# LI Inference Run Reference

- **Source Trained Model Folder**: single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LI_DnCNN_Attention
- **Target Dataset Folder**: MATLAB\A100_2p18e9_600km_70deg_30kHz

## Inference Performance Summary (LI)
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 2.426922e-15 | 8.805311 | 9.45 dB | 0.059681 |
| -5 | 1.066097e-15 | 3.803179 | 5.80 dB | 0.035014 |
| +0 | 3.831109e-16 | 1.400242 | 1.46 dB | 0.022347 |
| +5 | 1.308542e-16 | 0.448683 | -3.48 dB | 0.057126 |
| +10 | 3.771641e-17 | 0.139400 | -8.56 dB | 0.179018 |
| +15 | 1.007188e-17 | 0.040413 | -13.93 dB | 0.437110 |

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
