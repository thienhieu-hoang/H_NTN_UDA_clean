# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x3_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_stdFDA_minmaxReTrain`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.478983 | 0.461765 | 0.459236 |
| -5.0 | 0.442484 | 0.419056 | 0.415146 |
| 0.0 | 0.363592 | 0.339103 | 0.325323 |
| 5.0 | 0.249350 | 0.238237 | 0.211177 |
| 10.0 | 0.134006 | 0.154466 | 0.104709 |
| 15.0 | 0.051488 | 0.101966 | 0.035978 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 10.89 dB | -2.13 dB | -2.12 dB |
| -5.0 | 5.89 dB | -4.26 dB | -6.11 dB |
| 0.0 | 0.90 dB | -6.07 dB | -9.22 dB |
| 5.0 | -4.09 dB | -7.59 dB | -13.32 dB |
| 10.0 | -9.06 dB | -8.27 dB | -16.86 dB |
| 15.0 | -13.95 dB | -8.97 dB | -20.56 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0078 | 0.1643 | 0.0710 |
| -5.0 | 0.0480 | 0.3096 | 0.4364 |
| 0.0 | 0.1818 | 0.4427 | 0.6605 |
| 5.0 | 0.4265 | 0.5531 | 0.8283 |
| 10.0 | 0.6758 | 0.6184 | 0.9134 |
| 15.0 | 0.8415 | 0.6744 | 0.9574 |
