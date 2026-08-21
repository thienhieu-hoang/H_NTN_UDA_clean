# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\inference\DUR100__A100_2p18e9_600km_30kHz_DnCNN_ResNet_Attention_LI`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.456765 | 0.467171 | 0.450610 |
| -5.0 | 0.409113 | 0.423771 | 0.403962 |
| 0.0 | 0.323676 | 0.323893 | 0.316637 |
| 5.0 | 0.210719 | 0.212381 | 0.204220 |
| 10.0 | 0.102848 | 0.107805 | 0.098163 |
| 15.0 | 0.033314 | 0.039945 | 0.031162 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | -0.02 dB | 0.04 dB | -7.76 dB |
| -5.0 | -4.97 dB | -2.99 dB | -12.07 dB |
| 0.0 | -10.05 dB | -9.77 dB | -16.22 dB |
| 5.0 | -14.99 dB | -13.78 dB | -21.14 dB |
| 10.0 | -19.98 dB | -17.13 dB | -26.60 dB |
| 15.0 | -24.79 dB | -19.30 dB | -31.37 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.2531 | -0.2399 | 0.5560 |
| -5.0 | 0.5146 | -0.1924 | 0.8201 |
| 0.0 | 0.7605 | 0.6523 | 0.9325 |
| 5.0 | 0.8891 | 0.8584 | 0.9792 |
| 10.0 | 0.9626 | 0.9417 | 0.9933 |
| 15.0 | 0.9850 | 0.9632 | 0.9974 |
