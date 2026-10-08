# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x3_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_rms`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.478983 | 0.472107 | 0.459236 |
| -5.0 | 0.442484 | 0.435277 | 0.415146 |
| 0.0 | 0.363592 | 0.375334 | 0.325323 |
| 5.0 | 0.249350 | 0.291516 | 0.211177 |
| 10.0 | 0.134006 | 0.196331 | 0.104709 |
| 15.0 | 0.051488 | 0.129955 | 0.035978 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 10.89 dB | 0.85 dB | -2.12 dB |
| -5.0 | 5.89 dB | 1.16 dB | -6.11 dB |
| 0.0 | 0.90 dB | 0.13 dB | -9.22 dB |
| 5.0 | -4.09 dB | -2.19 dB | -13.32 dB |
| 10.0 | -9.06 dB | -5.06 dB | -16.86 dB |
| 15.0 | -13.95 dB | -6.67 dB | -20.56 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0078 | 0.0292 | 0.0710 |
| -5.0 | 0.0480 | 0.1743 | 0.4364 |
| 0.0 | 0.1818 | 0.2599 | 0.6605 |
| 5.0 | 0.4265 | 0.4140 | 0.8283 |
| 10.0 | 0.6758 | 0.5543 | 0.9134 |
| 15.0 | 0.8415 | 0.6408 | 0.9574 |
