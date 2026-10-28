# FDA Test-Time Inference Summary

- **FDA Approach**: `Approach 3.3`
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
| -10 | 5.365878 | 7.30 | 1.673678e-18 | 0.0488 | 0.0008 |
| -5 | 3.164642 | 5.00 | 9.410383e-19 | 0.1491 | 0.0008 |
| 0 | 1.670661 | 2.23 | 9.574636e-19 | 0.3113 | 0.0008 |
| 5 | 1.930416 | 2.86 | 6.204136e-19 | 0.4159 | 0.0008 |
| 10 | 0.923685 | -0.34 | 3.427133e-19 | 0.4847 | 0.0008 |
| 15 | 0.564810 | -2.48 | 2.605215e-19 | 0.5186 | 0.0008 |
