# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference_withFDA/A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_pseudo_app3p2_post_13x3`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LS+Transformer Pseudo Model FDA 3.2 (Post) | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.478983 | 0.477847 | 0.459236 |
| -5.0 | 0.442484 | 0.432547 | 0.415146 |
| 0.0 | 0.363592 | 0.350038 | 0.325323 |
| 5.0 | 0.249350 | 0.245581 | 0.211177 |
| 10.0 | 0.134006 | 0.149275 | 0.104709 |
| 15.0 | 0.051488 | 0.085291 | 0.035978 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LS+Transformer Pseudo Model FDA 3.2 (Post) | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 10.89 dB | 6.56 dB | -2.12 dB |
| -5.0 | 5.89 dB | 1.57 dB | -6.11 dB |
| 0.0 | 0.90 dB | -2.88 dB | -9.22 dB |
| 5.0 | -4.09 dB | -6.23 dB | -13.32 dB |
| 10.0 | -9.06 dB | -8.54 dB | -16.86 dB |
| 15.0 | -13.95 dB | -10.08 dB | -20.56 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LS+Transformer Pseudo Model FDA 3.2 (Post) | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0078 | 0.0032 | 0.0710 |
| -5.0 | 0.0480 | 0.1115 | 0.4364 |
| 0.0 | 0.1818 | 0.3116 | 0.6605 |
| 5.0 | 0.4265 | 0.5135 | 0.8283 |
| 10.0 | 0.6758 | 0.6609 | 0.9134 |
| 15.0 | 0.8415 | 0.7530 | 0.9574 |
