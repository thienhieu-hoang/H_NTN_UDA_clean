# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_rms`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.478983 | 0.472507 | 0.459236 |
| -5.0 | 0.442484 | 0.439319 | 0.415146 |
| 0.0 | 0.363592 | 0.374489 | 0.325323 |
| 5.0 | 0.249350 | 0.284281 | 0.211177 |
| 10.0 | 0.134006 | 0.201529 | 0.104709 |
| 15.0 | 0.051488 | 0.141610 | 0.035978 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 10.89 dB | 0.92 dB | -2.12 dB |
| -5.0 | 5.89 dB | 2.42 dB | -6.11 dB |
| 0.0 | 0.90 dB | -0.07 dB | -9.22 dB |
| 5.0 | -4.09 dB | -2.86 dB | -13.32 dB |
| 10.0 | -9.06 dB | -4.67 dB | -16.86 dB |
| 15.0 | -13.95 dB | -6.03 dB | -20.56 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0078 | 0.0485 | 0.0710 |
| -5.0 | 0.0480 | 0.1298 | 0.4364 |
| 0.0 | 0.1818 | 0.2649 | 0.6605 |
| 5.0 | 0.4265 | 0.3715 | 0.8283 |
| 10.0 | 0.6758 | 0.5096 | 0.9134 |
| 15.0 | 0.8415 | 0.5730 | 0.9574 |
