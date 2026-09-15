# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100port2__DUR100port2_pos3_2p18e9_600kmm_30kHz/LS_Attention_standardize`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.479813 | 0.451803 | 0.449371 |
| -5.0 | 0.442879 | 0.405031 | 0.403497 |
| 0.0 | 0.360976 | 0.314777 | 0.307726 |
| 5.0 | 0.238224 | 0.197893 | 0.189539 |
| 10.0 | 0.117123 | 0.091597 | 0.079230 |
| 15.0 | 0.032393 | 0.027739 | 0.014739 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 10.03 dB | -4.40 dB | -3.86 dB |
| -5.0 | 5.03 dB | -6.40 dB | -9.54 dB |
| 0.0 | 0.03 dB | -9.47 dB | -13.28 dB |
| 5.0 | -4.97 dB | -12.69 dB | -16.99 dB |
| 10.0 | -9.97 dB | -14.89 dB | -20.76 dB |
| 15.0 | -14.96 dB | -16.36 dB | -24.73 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0043 | 0.1614 | 0.1011 |
| -5.0 | 0.0259 | 0.2621 | 0.2852 |
| 0.0 | 0.1036 | 0.4006 | 0.5709 |
| 5.0 | 0.2599 | 0.5395 | 0.7537 |
| 10.0 | 0.4711 | 0.6690 | 0.8666 |
| 15.0 | 0.6764 | 0.7448 | 0.9347 |
