# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100port2__DUR100port2_pos3_2p18e9_600kmm_30kHz/syn_1`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LS+Transformer Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100port2__DUR100port2_pos3_2p18e9_600kmm_30kHz/LS_Attention` |
| 2 | **LS+Transformer Std Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100port2__DUR100port2_pos3_2p18e9_600kmm_30kHz/LS_Attention_standardize` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Transformer Inferred | LS+Transformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -3.51 dB | -4.40 dB | 10.03 dB | -3.86 dB |
| -5.0 | -6.67 dB | -6.40 dB | 5.03 dB | -9.54 dB |
| 0.0 | -9.54 dB | -9.47 dB | 0.03 dB | -13.28 dB |
| 5.0 | -12.18 dB | -12.69 dB | -4.97 dB | -16.99 dB |
| 10.0 | -13.90 dB | -14.89 dB | -9.97 dB | -20.76 dB |
| 15.0 | -15.85 dB | -16.36 dB | -14.96 dB | -24.73 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Transformer Inferred | LS+Transformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1273 | 0.1614 | 0.0043 | 0.1011 |
| -5.0 | 0.2667 | 0.2621 | 0.0259 | 0.2852 |
| 0.0 | 0.4001 | 0.4006 | 0.1036 | 0.5709 |
| 5.0 | 0.5319 | 0.5395 | 0.2599 | 0.7537 |
| 10.0 | 0.6494 | 0.6690 | 0.4711 | 0.8666 |
| 15.0 | 0.7337 | 0.7448 | 0.6764 | 0.9347 |

### C. MSE Comparison Table
| SNR (dB) | LS+Transformer Inferred | LS+Transformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.106e-19 | 9.188e-20 | 2.644e-18 | 1.036e-19 |
| -5.0 | 5.191e-20 | 5.552e-20 | 7.915e-19 | 2.463e-20 |
| 0.0 | 2.682e-20 | 2.756e-20 | 2.591e-19 | 1.015e-20 |
| 5.0 | 1.409e-20 | 1.224e-20 | 7.830e-20 | 4.206e-21 |
| 10.0 | 8.741e-21 | 6.789e-21 | 2.511e-20 | 1.794e-21 |
| 15.0 | 5.600e-21 | 4.896e-21 | 7.986e-21 | 7.590e-22 |

### D. BER Comparison Table
| SNR (dB) | LS+Transformer Inferred | LS+Transformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.452326 | 0.451803 | 0.479813 | 0.449371 |
| -5.0 | 0.404182 | 0.405031 | 0.442879 | 0.403497 |
| 0.0 | 0.314588 | 0.314777 | 0.360976 | 0.307726 |
| 5.0 | 0.199119 | 0.197893 | 0.238224 | 0.189539 |
| 10.0 | 0.095504 | 0.091597 | 0.117123 | 0.079230 |
| 15.0 | 0.029403 | 0.027739 | 0.032393 | 0.014739 |

