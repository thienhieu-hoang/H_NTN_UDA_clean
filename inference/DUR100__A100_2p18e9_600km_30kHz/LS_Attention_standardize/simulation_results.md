# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\LS_Attention_standardize`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.478966 | 0.447245 | 0.449289 |
| -5.0 | 0.440834 | 0.398096 | 0.403469 |
| 0.0 | 0.357410 | 0.306589 | 0.307729 |
| 5.0 | 0.234593 | 0.188677 | 0.189900 |
| 10.0 | 0.115154 | 0.077799 | 0.079528 |
| 15.0 | 0.032287 | 0.014410 | 0.014859 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 10.90 dB | -9.67 dB | -3.90 dB |
| -5.0 | 5.90 dB | -12.94 dB | -9.39 dB |
| 0.0 | 0.90 dB | -15.55 dB | -13.25 dB |
| 5.0 | -4.10 dB | -19.16 dB | -17.00 dB |
| 10.0 | -9.10 dB | -22.73 dB | -20.76 dB |
| 15.0 | -14.09 dB | -25.89 dB | -24.78 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0028 | 0.5340 | 0.1001 |
| -5.0 | 0.0208 | 0.6966 | 0.2817 |
| 0.0 | 0.0818 | 0.7663 | 0.5681 |
| 5.0 | 0.2261 | 0.8586 | 0.7547 |
| 10.0 | 0.4422 | 0.9298 | 0.8665 |
| 15.0 | 0.6533 | 0.9631 | 0.9350 |
