# FDA Test-Time Inference Summary

- **FDA Approach**: `Approach 3.2`
- **FDA Stage Legacy**: `dual_pcp`
- **Window Type**: `tukey_outward`
- **Window Geometry**: Delay = 13 px, Doppler = 3 px
- **Taper Setting**: Delay taper = 2 rows, Doppler taper = 1 cols
- **Pilot Consistency**: Alpha = 0.5
- **Model Directory**: `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\pseudo_A100Perfect_DUR300LSAttention_rms_13x3\LS_Attention_rmsFDA_rmsReTrain`
- **Dataset Directory**: `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps`
- **Normalization**: `rms`

## Evaluation Metrics

| SNR (dB) | NMSE | NMSE (dB) | MMSE | SSIM | Avg Time/Sample (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| -10 | 4.096639 | 6.12 | 1.277788e-18 | 0.0562 | 0.0006 |
| -5 | 1.393655 | 1.44 | 4.144175e-19 | 0.2243 | 0.0006 |
| 0 | 0.414885 | -3.82 | 2.377727e-19 | 0.4217 | 0.0006 |
| 5 | 0.246562 | -6.08 | 7.924226e-20 | 0.5905 | 0.0006 |
| 10 | 0.070191 | -11.54 | 2.604288e-20 | 0.7228 | 0.0006 |
| 15 | 0.075304 | -11.23 | 3.473413e-20 | 0.7994 | 0.0006 |
