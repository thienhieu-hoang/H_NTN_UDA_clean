# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100p1__DUR100p2_2p18e9_600km_30kHz/LS_Attention_AxialAttention_standardize`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.479010 | 0.451533 | 0.449085 |
| -5.0 | 0.441215 | 0.404302 | 0.403460 |
| 0.0 | 0.357563 | 0.311965 | 0.307659 |
| 5.0 | 0.233879 | 0.195615 | 0.189585 |
| 10.0 | 0.113318 | 0.089103 | 0.078925 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 9.53 dB | -4.27 dB | -3.92 dB |
| -5.0 | 4.53 dB | -6.79 dB | -9.56 dB |
| 0.0 | -0.47 dB | -10.56 dB | -13.34 dB |
| 5.0 | -5.47 dB | -13.40 dB | -17.01 dB |
| 10.0 | -10.47 dB | -15.34 dB | -20.94 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0056 | 0.1534 | 0.1016 |
| -5.0 | 0.0298 | 0.2589 | 0.2864 |
| 0.0 | 0.1147 | 0.4153 | 0.5707 |
| 5.0 | 0.2775 | 0.5532 | 0.7570 |
| 10.0 | 0.4928 | 0.6715 | 0.8719 |
