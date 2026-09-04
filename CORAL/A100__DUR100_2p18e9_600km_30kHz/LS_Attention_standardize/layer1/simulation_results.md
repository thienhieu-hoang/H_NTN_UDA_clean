# CORAL UDA Channel Equalization & Domain Performance Summary

**Batch Directory:** `c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/CORAL/A100__DUR100_2p18e9_600km_30kHz/LS_Attention_standardize/layer1`  
**Model Evaluation:** `LS+Attention(Std) Inferred`  

## Multi-Domain BER Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.456137 | 0.451271 | 0.479818 | 0.449321 |
| -5.0 | 0.408746 | 0.403240 | 0.441466 | 0.403807 |
| 0.0 | 0.323213 | 0.311546 | 0.357242 | 0.306450 |
| 5.0 | 0.209523 | 0.195093 | 0.233443 | 0.188221 |
| 10.0 | 0.106542 | 0.090398 | 0.113611 | 0.078185 |
| 15.0 | 0.034405 | 0.027974 | 0.030859 | 0.013600 |

## Multi-Domain NMSE (dB) Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -4.30 dB | -4.83 dB | 10.73 dB | -3.78 dB |
| -5.0 | -7.09 dB | -7.56 dB | 5.73 dB | -9.27 dB |
| 0.0 | -10.93 dB | -10.62 dB | 0.74 dB | -13.42 dB |
| 5.0 | -14.90 dB | -13.55 dB | -4.26 dB | -17.23 dB |
| 10.0 | -18.34 dB | -15.05 dB | -9.26 dB | -20.80 dB |
| 15.0 | -22.02 dB | -16.06 dB | -14.27 dB | -24.92 dB |

## Multi-Domain SSIM Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.3812 | 0.1552 | 0.0026 | 0.0897 |
| -5.0 | 0.5408 | 0.3140 | 0.0229 | 0.2984 |
| 0.0 | 0.8012 | 0.4342 | 0.0826 | 0.5559 |
| 5.0 | 0.9011 | 0.5629 | 0.2211 | 0.7456 |
| 10.0 | 0.9552 | 0.6650 | 0.4289 | 0.8626 |
| 15.0 | 0.9758 | 0.7295 | 0.6602 | 0.9380 |
