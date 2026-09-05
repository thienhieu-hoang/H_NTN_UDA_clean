# CORAL UDA Channel Equalization & Domain Performance Summary

**Batch Directory:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\CORAL\A100__DUR100_2p18e9_600km_30kHz\LI_DnCNN_pHead\block3`  
**Model Evaluation:** `LI+DnCNN CORAL pHead (block3)`  

## Multi-Domain BER Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.459552 | 0.484037 | 0.479818 | 0.449321 |
| -5.0 | 0.413578 | 0.467753 | 0.441466 | 0.403807 |
| 0.0 | 0.327744 | 0.431461 | 0.357242 | 0.306450 |
| 5.0 | 0.228450 | 0.372638 | 0.233443 | 0.188221 |
| 10.0 | 0.121582 | 0.304169 | 0.113611 | 0.078185 |
| 15.0 | 0.042231 | 0.233798 | 0.030859 | 0.013600 |

## Multi-Domain NMSE (dB) Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -2.13 dB | 13.40 dB | 10.73 dB | -3.78 dB |
| -5.0 | -5.36 dB | 10.81 dB | 5.73 dB | -9.27 dB |
| 0.0 | -8.87 dB | 7.02 dB | 0.74 dB | -13.42 dB |
| 5.0 | -8.39 dB | 3.16 dB | -4.26 dB | -17.23 dB |
| 10.0 | -12.44 dB | -0.02 dB | -9.26 dB | -20.80 dB |
| 15.0 | -17.00 dB | -2.43 dB | -14.27 dB | -24.92 dB |

## Multi-Domain SSIM Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1831 | -0.0043 | 0.0026 | 0.0897 |
| -5.0 | 0.3778 | -0.0042 | 0.0229 | 0.2984 |
| 0.0 | 0.6833 | 0.0098 | 0.0826 | 0.5559 |
| 5.0 | 0.6478 | -0.0610 | 0.2211 | 0.7456 |
| 10.0 | 0.8389 | -0.0553 | 0.4289 | 0.8626 |
| 15.0 | 0.9300 | -0.0135 | 0.6602 | 0.9380 |
