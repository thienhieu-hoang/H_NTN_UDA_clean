# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LS_Attention_AxialAttention`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.478966 | 0.453147 | 0.449289 |
| -5.0 | 0.440834 | 0.405290 | 0.403469 |
| 0.0 | 0.357410 | 0.314550 | 0.307729 |
| 5.0 | 0.234593 | 0.199456 | 0.189900 |
| 10.0 | 0.115154 | 0.092308 | 0.079528 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 10.90 dB | -3.01 dB | -3.90 dB |
| -5.0 | 5.90 dB | -6.22 dB | -9.39 dB |
| 0.0 | 0.90 dB | -9.49 dB | -13.25 dB |
| 5.0 | -4.10 dB | -12.09 dB | -17.00 dB |
| 10.0 | -9.10 dB | -14.70 dB | -20.76 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0028 | 0.0998 | 0.1001 |
| -5.0 | 0.0208 | 0.2511 | 0.2817 |
| 0.0 | 0.0818 | 0.3976 | 0.5681 |
| 5.0 | 0.2261 | 0.5150 | 0.7547 |
| 10.0 | 0.4422 | 0.6508 | 0.8665 |
