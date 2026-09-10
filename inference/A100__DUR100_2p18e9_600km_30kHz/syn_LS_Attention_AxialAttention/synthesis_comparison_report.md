# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/syn_1`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LS+Transformer+AxialTransformer Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LS_Attention_AxialAttention` |
| 2 | **LS+Transformer+AxialTransformer Std Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LS_Attention_AxialAttention_standardize` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Transformer+AxialTransformer Inferred | LS+Transformer+AxialTransformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -3.01 dB | -4.20 dB | 10.90 dB | -3.90 dB |
| -5.0 | -6.22 dB | -6.76 dB | 5.90 dB | -9.39 dB |
| 0.0 | -9.49 dB | -10.44 dB | 0.90 dB | -13.25 dB |
| 5.0 | -12.09 dB | -13.28 dB | -4.10 dB | -17.00 dB |
| 10.0 | -14.70 dB | -15.33 dB | -9.10 dB | -20.76 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Transformer+AxialTransformer Inferred | LS+Transformer+AxialTransformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.0998 | 0.1427 | 0.0028 | 0.1001 |
| -5.0 | 0.2511 | 0.2659 | 0.0208 | 0.2817 |
| 0.0 | 0.3976 | 0.4129 | 0.0818 | 0.5681 |
| 5.0 | 0.5150 | 0.5509 | 0.2261 | 0.7547 |
| 10.0 | 0.6508 | 0.6721 | 0.4422 | 0.8665 |

### C. MSE Comparison Table
| SNR (dB) | LS+Transformer+AxialTransformer Inferred | LS+Transformer+AxialTransformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.238e-19 | 9.637e-20 | 3.145e-18 | 1.009e-19 |
| -5.0 | 5.995e-20 | 5.271e-20 | 9.961e-19 | 2.540e-20 |
| 0.0 | 2.714e-20 | 2.149e-20 | 3.122e-19 | 1.047e-20 |
| 5.0 | 1.414e-20 | 1.046e-20 | 9.591e-20 | 4.308e-21 |
| 10.0 | 7.200e-21 | 6.216e-21 | 2.949e-20 | 1.793e-21 |

### D. BER Comparison Table
| SNR (dB) | LS+Transformer+AxialTransformer Inferred | LS+Transformer+AxialTransformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.453147 | 0.451618 | 0.478966 | 0.449289 |
| -5.0 | 0.405290 | 0.404234 | 0.440834 | 0.403469 |
| 0.0 | 0.314550 | 0.312188 | 0.357410 | 0.307729 |
| 5.0 | 0.199456 | 0.196057 | 0.234593 | 0.189900 |
| 10.0 | 0.092308 | 0.089724 | 0.115154 | 0.079528 |

