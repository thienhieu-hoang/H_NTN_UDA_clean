# CORAL UDA Channel Equalization & Domain Performance Summary

**Batch Directory:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\CORAL\A100__DUR100_2p18e9_600km_30kHz\LS_Attention_pHead_standardize\layer1_layer2`  
**Model Evaluation:** `LS+Transformer CORAL pHead (layer1_layer2)`  

## Multi-Domain BER Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.455745 | 0.449507 | 0.479818 | 0.449321 |
| -5.0 | 0.407842 | 0.402474 | 0.441466 | 0.403807 |
| 0.0 | 0.324605 | 0.309950 | 0.357242 | 0.306450 |
| 5.0 | 0.209459 | 0.195872 | 0.233443 | 0.188221 |
| 10.0 | 0.105722 | 0.089879 | 0.113611 | 0.078185 |
| 15.0 | 0.034212 | 0.025605 | 0.030859 | 0.013600 |

## Multi-Domain NMSE (dB) Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -4.58 dB | -6.12 dB | 10.73 dB | -3.78 dB |
| -5.0 | -7.82 dB | -8.13 dB | 5.73 dB | -9.27 dB |
| 0.0 | -10.19 dB | -11.32 dB | 0.74 dB | -13.42 dB |
| 5.0 | -14.74 dB | -13.32 dB | -4.26 dB | -17.23 dB |
| 10.0 | -18.45 dB | -15.28 dB | -9.26 dB | -20.80 dB |
| 15.0 | -21.85 dB | -16.89 dB | -14.27 dB | -24.92 dB |

## Multi-Domain SSIM Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.2811 | 0.1496 | 0.0026 | 0.0897 |
| -5.0 | 0.5707 | 0.3424 | 0.0229 | 0.2984 |
| 0.0 | 0.7434 | 0.4462 | 0.0826 | 0.5559 |
| 5.0 | 0.8966 | 0.5629 | 0.2211 | 0.7456 |
| 10.0 | 0.9532 | 0.6689 | 0.4289 | 0.8626 |
| 15.0 | 0.9756 | 0.7530 | 0.6602 | 0.9380 |
