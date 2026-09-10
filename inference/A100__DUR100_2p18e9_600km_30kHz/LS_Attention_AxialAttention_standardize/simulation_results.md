# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LS_Attention_AxialAttention_standardize`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.478966 | 0.451618 | 0.449289 |
| -5.0 | 0.440834 | 0.404234 | 0.403469 |
| 0.0 | 0.357410 | 0.312188 | 0.307729 |
| 5.0 | 0.234593 | 0.196057 | 0.189900 |
| 10.0 | 0.115154 | 0.089724 | 0.079528 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 10.90 dB | -4.20 dB | -3.90 dB |
| -5.0 | 5.90 dB | -6.76 dB | -9.39 dB |
| 0.0 | 0.90 dB | -10.44 dB | -13.25 dB |
| 5.0 | -4.10 dB | -13.28 dB | -17.00 dB |
| 10.0 | -9.10 dB | -15.33 dB | -20.76 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0028 | 0.1427 | 0.1001 |
| -5.0 | 0.0208 | 0.2659 | 0.2817 |
| 0.0 | 0.0818 | 0.4129 | 0.5681 |
| 5.0 | 0.2261 | 0.5509 | 0.7547 |
| 10.0 | 0.4422 | 0.6721 | 0.8665 |
