# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\LS_Attention_standardize`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.482047 | 0.456013 | 0.454276 |
| -5.0 | 0.449682 | 0.411688 | 0.411011 |
| 0.0 | 0.380375 | 0.327831 | 0.320705 |
| 5.0 | 0.273889 | 0.220744 | 0.207732 |
| 10.0 | 0.160101 | 0.122571 | 0.101399 |
| 15.0 | 0.068865 | 0.056730 | 0.032343 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 13.15 dB | -3.83 dB | -2.69 dB |
| -5.0 | 8.21 dB | -6.32 dB | -7.36 dB |
| 0.0 | 3.25 dB | -8.64 dB | -11.91 dB |
| 5.0 | -1.84 dB | -10.67 dB | -16.38 dB |
| 10.0 | -6.79 dB | -12.63 dB | -20.72 dB |
| 15.0 | -11.82 dB | -14.26 dB | -25.13 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0047 | 0.2315 | 0.1009 |
| -5.0 | 0.0248 | 0.4338 | 0.5012 |
| 0.0 | 0.1196 | 0.6584 | 0.8124 |
| 5.0 | 0.3379 | 0.7756 | 0.9283 |
| 10.0 | 0.6121 | 0.8584 | 0.9714 |
| 15.0 | 0.8087 | 0.9031 | 0.9885 |
