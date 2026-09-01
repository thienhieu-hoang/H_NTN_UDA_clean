# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LI_DnCNN_AxialAttention`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.478966 | 0.480875 | 0.449289 |
| -5.0 | 0.440834 | 0.446573 | 0.403469 |
| 0.0 | 0.357410 | 0.371394 | 0.307729 |
| 5.0 | 0.234593 | 0.245172 | 0.189900 |
| 10.0 | 0.115154 | 0.115642 | 0.079528 |
| 15.0 | 0.032287 | 0.032403 | 0.014859 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 10.90 dB | 12.67 dB | -3.90 dB |
| -5.0 | 5.90 dB | 7.45 dB | -9.39 dB |
| 0.0 | 0.90 dB | 3.43 dB | -13.25 dB |
| 5.0 | -4.10 dB | -2.84 dB | -17.00 dB |
| 10.0 | -9.10 dB | -8.72 dB | -20.76 dB |
| 15.0 | -14.09 dB | -13.97 dB | -24.78 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0028 | 0.0047 | 0.1001 |
| -5.0 | 0.0208 | 0.0263 | 0.2817 |
| 0.0 | 0.0818 | 0.0900 | 0.5681 |
| 5.0 | 0.2261 | 0.2503 | 0.7547 |
| 10.0 | 0.4422 | 0.4493 | 0.8665 |
| 15.0 | 0.6533 | 0.6637 | 0.9350 |
