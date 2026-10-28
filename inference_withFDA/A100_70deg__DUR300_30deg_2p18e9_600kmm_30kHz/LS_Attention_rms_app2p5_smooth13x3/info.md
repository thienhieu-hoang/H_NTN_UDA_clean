# FDA Test-Time Inference Summary

- **FDA Approach**: `Approach 2.5`
- **FDA Stage Legacy**: `dual_pcp`
- **Window Type**: `tukey_outward`
- **Window Geometry**: Delay = 13 px, Doppler = 3 px
- **Taper Setting**: Delay taper = 2 rows, Doppler taper = 1 cols
- **Pilot Consistency**: Alpha = 0.5
- **Model Directory**: `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_rms`
- **Dataset Directory**: `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps`
- **Normalization**: `rms`

## Evaluation Metrics

| SNR (dB) | NMSE | NMSE (dB) | MMSE | SSIM | Avg Time/Sample (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| -10 | 3.529572 | 5.48 | 1.100913e-18 | 0.0499 | 0.0017 |
| -5 | 1.511040 | 1.79 | 4.493232e-19 | 0.1079 | 0.0017 |
| 0 | 0.412991 | -3.84 | 2.366873e-19 | 0.1551 | 0.0017 |
| 5 | 0.307115 | -5.13 | 9.870337e-20 | 0.1484 | 0.0017 |
| 10 | 0.153729 | -8.13 | 5.703776e-20 | 0.1562 | 0.0018 |
| 15 | 0.174082 | -7.59 | 8.029619e-20 | 0.1465 | 0.0018 |
