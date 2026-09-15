# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100p1__DUR100p2_2p18e9_600km_30kHz/syn_1`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LS+Transformer+AxialTransformer Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100p1__DUR100p2_2p18e9_600km_30kHz/LS_Attention_AxialAttention` |
| 2 | **LS+Transformer+AxialTransformer Std Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100p1__DUR100p2_2p18e9_600km_30kHz/LS_Attention_AxialAttention_standardize` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Transformer+AxialTransformer Inferred | LS+Transformer+AxialTransformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -3.08 dB | -4.27 dB | 9.53 dB | -3.92 dB |
| -5.0 | -6.22 dB | -6.79 dB | 4.53 dB | -9.56 dB |
| 0.0 | -9.60 dB | -10.56 dB | -0.47 dB | -13.34 dB |
| 5.0 | -12.14 dB | -13.40 dB | -5.47 dB | -17.01 dB |
| 10.0 | -14.67 dB | -15.34 dB | -10.47 dB | -20.94 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Transformer+AxialTransformer Inferred | LS+Transformer+AxialTransformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1182 | 0.1534 | 0.0056 | 0.1016 |
| -5.0 | 0.2410 | 0.2589 | 0.0298 | 0.2864 |
| 0.0 | 0.3952 | 0.4153 | 0.1147 | 0.5707 |
| 5.0 | 0.5186 | 0.5532 | 0.2775 | 0.7570 |
| 10.0 | 0.6441 | 0.6715 | 0.4928 | 0.8719 |

### C. MSE Comparison Table
| SNR (dB) | LS+Transformer+AxialTransformer Inferred | LS+Transformer+AxialTransformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.131e-19 | 8.731e-20 | 2.155e-18 | 9.503e-20 |
| -5.0 | 5.945e-20 | 5.162e-20 | 7.157e-19 | 2.451e-20 |
| 0.0 | 2.644e-20 | 2.078e-20 | 2.250e-19 | 9.829e-21 |
| 5.0 | 1.433e-20 | 1.025e-20 | 7.115e-20 | 4.226e-21 |
| 10.0 | 7.401e-21 | 6.214e-21 | 2.228e-20 | 1.790e-21 |

### D. BER Comparison Table
| SNR (dB) | LS+Transformer+AxialTransformer Inferred | LS+Transformer+AxialTransformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.453221 | 0.451533 | 0.479010 | 0.449085 |
| -5.0 | 0.405297 | 0.404302 | 0.441215 | 0.403460 |
| 0.0 | 0.314415 | 0.311965 | 0.357563 | 0.307659 |
| 5.0 | 0.199038 | 0.195615 | 0.233879 | 0.189585 |
| 10.0 | 0.091837 | 0.089103 | 0.113318 | 0.078925 |

