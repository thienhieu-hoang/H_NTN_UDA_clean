# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference_withFDA/A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_pseudo_app3p3_dual_13x3`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LS+Transformer Pseudo Model FDA 3.3 (Dual) | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.478983 | 0.481424 | 0.459236 |
| -5.0 | 0.442484 | 0.443967 | 0.415146 |
| 0.0 | 0.363592 | 0.373580 | 0.325323 |
| 5.0 | 0.249350 | 0.292779 | 0.211177 |
| 10.0 | 0.134006 | 0.226570 | 0.104709 |
| 15.0 | 0.051488 | 0.191066 | 0.035978 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LS+Transformer Pseudo Model FDA 3.3 (Dual) | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 10.89 dB | 7.14 dB | -2.12 dB |
| -5.0 | 5.89 dB | 11.09 dB | -6.11 dB |
| 0.0 | 0.90 dB | 8.86 dB | -9.22 dB |
| 5.0 | -4.09 dB | 21.63 dB | -13.32 dB |
| 10.0 | -9.06 dB | 18.78 dB | -16.86 dB |
| 15.0 | -13.95 dB | 17.63 dB | -20.56 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LS+Transformer Pseudo Model FDA 3.3 (Dual) | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0078 | 0.0033 | 0.0710 |
| -5.0 | 0.0480 | 0.0668 | 0.4364 |
| 0.0 | 0.1818 | 0.2028 | 0.6605 |
| 5.0 | 0.4265 | 0.3060 | 0.8283 |
| 10.0 | 0.6758 | 0.3604 | 0.9134 |
| 15.0 | 0.8415 | 0.3996 | 0.9574 |
