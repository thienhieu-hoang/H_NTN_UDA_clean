# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100p1__DUR100p2_2p18e9_600km_30kHz/LS_Attention`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.479010 | 0.452254 | 0.449085 |
| -5.0 | 0.441215 | 0.404407 | 0.403460 |
| 0.0 | 0.357563 | 0.314558 | 0.307659 |
| 5.0 | 0.233879 | 0.199239 | 0.189585 |
| 10.0 | 0.113318 | 0.094576 | 0.078925 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 9.53 dB | -3.38 dB | -3.92 dB |
| -5.0 | 4.53 dB | -6.69 dB | -9.56 dB |
| 0.0 | -0.47 dB | -9.56 dB | -13.34 dB |
| 5.0 | -5.47 dB | -12.11 dB | -17.01 dB |
| 10.0 | -10.47 dB | -14.05 dB | -20.94 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0056 | 0.1284 | 0.1016 |
| -5.0 | 0.0298 | 0.2550 | 0.2864 |
| 0.0 | 0.1147 | 0.4011 | 0.5707 |
| 5.0 | 0.2775 | 0.5337 | 0.7570 |
| 10.0 | 0.4928 | 0.6536 | 0.8719 |
