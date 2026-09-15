# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100p1__DUR100p2_2p18e9_600km_30kHz/syn_2`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LS+Transformer Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100p1__DUR100p2_2p18e9_600km_30kHz/LS_Attention` |
| 2 | **LS+Transformer Std Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100p1__DUR100p2_2p18e9_600km_30kHz/LS_Attention_standardize` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Transformer Inferred | LS+Transformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -3.38 dB | -4.35 dB | 9.53 dB | -3.92 dB |
| -5.0 | -6.69 dB | -6.32 dB | 4.53 dB | -9.56 dB |
| 0.0 | -9.56 dB | -9.49 dB | -0.47 dB | -13.34 dB |
| 5.0 | -12.11 dB | -12.68 dB | -5.47 dB | -17.01 dB |
| 10.0 | -14.05 dB | -14.93 dB | -10.47 dB | -20.94 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Transformer Inferred | LS+Transformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1284 | 0.1518 | 0.0056 | 0.1016 |
| -5.0 | 0.2550 | 0.2484 | 0.0298 | 0.2864 |
| 0.0 | 0.4011 | 0.4034 | 0.1147 | 0.5707 |
| 5.0 | 0.5337 | 0.5395 | 0.2775 | 0.7570 |
| 10.0 | 0.6536 | 0.6696 | 0.4928 | 0.8719 |

### C. MSE Comparison Table
| SNR (dB) | LS+Transformer Inferred | LS+Transformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.045e-19 | 8.497e-20 | 2.155e-18 | 9.503e-20 |
| -5.0 | 5.351e-20 | 5.773e-20 | 7.157e-19 | 2.451e-20 |
| 0.0 | 2.720e-20 | 2.724e-20 | 2.250e-19 | 9.829e-21 |
| 5.0 | 1.437e-20 | 1.253e-20 | 7.115e-20 | 4.226e-21 |
| 10.0 | 8.619e-21 | 6.956e-21 | 2.228e-20 | 1.790e-21 |

### D. BER Comparison Table
| SNR (dB) | LS+Transformer Inferred | LS+Transformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.452254 | 0.451776 | 0.479010 | 0.449085 |
| -5.0 | 0.404407 | 0.405214 | 0.441215 | 0.403460 |
| 0.0 | 0.314558 | 0.314691 | 0.357563 | 0.307659 |
| 5.0 | 0.199239 | 0.197859 | 0.233879 | 0.189585 |
| 10.0 | 0.094576 | 0.091276 | 0.113318 | 0.078925 |

