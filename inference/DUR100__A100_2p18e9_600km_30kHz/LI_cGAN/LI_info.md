# LI Inference Run Reference

- **Source Trained Model Folder**: single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LI_cGAN
- **Target Dataset Folder**: MATLAB\A100_2p18e9_600km_70deg_30kHz

## Inference Performance Summary (LI)
| SNR (dB) | MMSE | NMSE | NMSE (dB) | SSIM |
|----------|------|------|-----------|------|
| -10 | 6.782976e-15 | 24.108405 | 13.82 dB | 0.126667 |
| -5 | 2.502742e-15 | 8.643960 | 9.37 dB | 0.118684 |
| +0 | 7.903475e-16 | 2.831134 | 4.52 dB | 0.126805 |
| +5 | 3.620527e-16 | 1.281231 | 1.08 dB | 0.196448 |
| +10 | 2.196632e-16 | 0.854703 | -0.68 dB | 0.281167 |
| +15 | 1.415199e-16 | 0.558498 | -2.53 dB | 0.369235 |

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
