# CORAL UDA Channel Equalization & Domain Performance Summary

**Batch Directory:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\CORAL\A100__DUR100_2p18e9_600km_30kHz\LS_Attention_pHead_standardize\layer1`  
**Model Evaluation:** `LS+Transformer CORAL pHead (layer1)`  

## Multi-Domain BER Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.455740 | 0.449852 | 0.479818 | 0.449321 |
| -5.0 | 0.412396 | 0.401364 | 0.441466 | 0.403807 |
| 0.0 | 0.323945 | 0.309030 | 0.357242 | 0.306450 |
| 5.0 | 0.209895 | 0.195151 | 0.233443 | 0.188221 |
| 10.0 | 0.106105 | 0.090045 | 0.113611 | 0.078185 |
| 15.0 | 0.034247 | 0.025521 | 0.030859 | 0.013600 |

## Multi-Domain NMSE (dB) Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -4.60 dB | -5.94 dB | 10.73 dB | -3.78 dB |
| -5.0 | -6.46 dB | -9.34 dB | 5.73 dB | -9.27 dB |
| 0.0 | -10.48 dB | -11.78 dB | 0.74 dB | -13.42 dB |
| 5.0 | -14.80 dB | -13.40 dB | -4.26 dB | -17.23 dB |
| 10.0 | -18.18 dB | -15.16 dB | -9.26 dB | -20.80 dB |
| 15.0 | -21.98 dB | -16.95 dB | -14.27 dB | -24.92 dB |

## Multi-Domain SSIM Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.3201 | 0.1696 | 0.0026 | 0.0897 |
| -5.0 | 0.3562 | 0.2652 | 0.0229 | 0.2984 |
| 0.0 | 0.7666 | 0.4604 | 0.0826 | 0.5559 |
| 5.0 | 0.8971 | 0.5479 | 0.2211 | 0.7456 |
| 10.0 | 0.9519 | 0.6678 | 0.4289 | 0.8626 |
| 15.0 | 0.9773 | 0.7494 | 0.6602 | 0.9380 |
