# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LIpseudo13x3_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_rms`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.478983 | 0.466537 | 0.459236 |
| -5.0 | 0.442484 | 0.426597 | 0.415146 |
| 0.0 | 0.363592 | 0.352597 | 0.325323 |
| 5.0 | 0.249350 | 0.256760 | 0.211177 |
| 10.0 | 0.134006 | 0.159715 | 0.104709 |
| 15.0 | 0.051488 | 0.082788 | 0.035978 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 10.89 dB | 1.49 dB | -2.12 dB |
| -5.0 | 5.89 dB | -0.02 dB | -6.11 dB |
| 0.0 | 0.90 dB | -1.98 dB | -9.22 dB |
| 5.0 | -4.09 dB | -5.15 dB | -13.32 dB |
| 10.0 | -9.06 dB | -8.31 dB | -16.86 dB |
| 15.0 | -13.95 dB | -10.91 dB | -20.56 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0078 | 0.1029 | 0.0710 |
| -5.0 | 0.0480 | 0.2301 | 0.4364 |
| 0.0 | 0.1818 | 0.4227 | 0.6605 |
| 5.0 | 0.4265 | 0.6192 | 0.8283 |
| 10.0 | 0.6758 | 0.7625 | 0.9134 |
| 15.0 | 0.8415 | 0.8471 | 0.9574 |
