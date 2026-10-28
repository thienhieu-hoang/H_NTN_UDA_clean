# FDA Test-Time Inference Summary

- **FDA Approach**: `Approach 2.6`
- **FDA Stage Legacy**: `dual_pcp`
- **Window Type**: `tukey_outward`
- **Window Geometry**: Delay = 13 px, Doppler = 3 px
- **Taper Setting**: Delay taper = 2 rows, Doppler taper = 1 cols
- **Pilot Consistency**: Alpha = 0.5
- **Model Directory**: `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_rms`
- **Model 2 Directory**: `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\pseudo_A100Perfect_DUR300LSAttention_rms_13x3\LS_Attention_rmsFDA_rmsReTrain`
- **Dataset Directory**: `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps`
- **Normalization**: `rms`

## Evaluation Metrics

| SNR (dB) | NMSE | NMSE (dB) | MMSE | SSIM | Avg Time/Sample (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| -10 | 4.075645 | 6.10 | 1.271240e-18 | 0.0406 | 0.0011 |
| -5 | 1.688901 | 2.28 | 5.022119e-19 | 0.1325 | 0.0011 |
| 0 | 0.499070 | -3.02 | 2.860192e-19 | 0.3351 | 0.0011 |
| 5 | 0.348189 | -4.58 | 1.119038e-19 | 0.5115 | 0.0011 |
| 10 | 0.125864 | -9.00 | 4.669893e-20 | 0.6317 | 0.0011 |
| 15 | 0.119828 | -9.21 | 5.527145e-20 | 0.6982 | 0.0011 |
