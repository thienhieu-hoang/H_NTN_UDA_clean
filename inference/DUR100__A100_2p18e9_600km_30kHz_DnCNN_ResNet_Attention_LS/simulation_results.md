# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\inference\DUR100__A100_2p18e9_600km_30kHz_DnCNN_ResNet_Attention_LS`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.456765 | 0.476461 | 0.450610 |
| -5.0 | 0.409113 | 0.428193 | 0.403962 |
| 0.0 | 0.323676 | 0.364798 | 0.316637 |
| 5.0 | 0.210719 | 0.248502 | 0.204220 |
| 10.0 | 0.102848 | 0.215019 | 0.098163 |
| 15.0 | 0.033314 | 0.157682 | 0.031162 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | -0.02 dB | -1.09 dB | -7.76 dB |
| -5.0 | -4.97 dB | -2.99 dB | -12.07 dB |
| 0.0 | -10.05 dB | -3.72 dB | -16.22 dB |
| 5.0 | -14.99 dB | -5.50 dB | -21.14 dB |
| 10.0 | -19.98 dB | -3.68 dB | -26.60 dB |
| 15.0 | -24.79 dB | -4.99 dB | -31.37 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.2531 | 0.0446 | 0.5560 |
| -5.0 | 0.5146 | 0.1790 | 0.8201 |
| 0.0 | 0.7605 | 0.3071 | 0.9325 |
| 5.0 | 0.8891 | 0.5000 | 0.9792 |
| 10.0 | 0.9626 | 0.4675 | 0.9933 |
| 15.0 | 0.9850 | 0.5317 | 0.9974 |
