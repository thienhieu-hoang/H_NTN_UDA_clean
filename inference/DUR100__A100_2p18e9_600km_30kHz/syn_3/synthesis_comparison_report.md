# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\syn_3`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LI+DnCNN+AxialAttention Inferred** | `C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\LI_DnCNN_AxialAttention` |
| 2 | **LI+DnCNN+CrossAttention Inferred** | `C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\LI_DnCNN_CrossAttention` |
| 3 | **LS+Attention Std Inferred** | `C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\LS_Attention_standardize` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+DnCNN+AxialAttention Inferred | LI+DnCNN+CrossAttention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 9.59 dB | 10.74 dB | -3.83 dB | 13.15 dB | -2.69 dB |
| -5.0 | 6.17 dB | 5.70 dB | -6.32 dB | 8.21 dB | -7.36 dB |
| 0.0 | 1.67 dB | 2.22 dB | -8.64 dB | 3.25 dB | -11.91 dB |
| 5.0 | -3.27 dB | -2.63 dB | -10.67 dB | -1.84 dB | -16.38 dB |
| 10.0 | -8.42 dB | -8.54 dB | -12.63 dB | -6.79 dB | -20.72 dB |
| 15.0 | -13.41 dB | -14.20 dB | -14.26 dB | -11.82 dB | -25.13 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+DnCNN+AxialAttention Inferred | LI+DnCNN+CrossAttention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -0.0050 | -0.0040 | 0.2315 | 0.0047 | 0.1009 |
| -5.0 | -0.0181 | -0.0232 | 0.4338 | 0.0248 | 0.5012 |
| 0.0 | 0.0026 | -0.0230 | 0.6584 | 0.1196 | 0.8124 |
| 5.0 | 0.2384 | 0.1854 | 0.7756 | 0.3379 | 0.9283 |
| 10.0 | 0.5901 | 0.6106 | 0.8584 | 0.6121 | 0.9714 |
| 15.0 | 0.8273 | 0.8501 | 0.9031 | 0.8087 | 0.9885 |

### C. MSE Comparison Table
| SNR (dB) | LI+DnCNN+AxialAttention Inferred | LI+DnCNN+CrossAttention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 2.525e-15 | 3.304e-15 | 9.978e-17 | 5.798e-15 | 1.406e-16 |
| -5.0 | 1.171e-15 | 1.049e-15 | 5.651e-17 | 1.924e-15 | 4.385e-17 |
| 0.0 | 4.115e-16 | 4.665e-16 | 2.986e-17 | 5.973e-16 | 1.397e-17 |
| 5.0 | 1.370e-16 | 1.619e-16 | 1.775e-17 | 1.938e-16 | 5.567e-18 |
| 10.0 | 3.844e-17 | 3.837e-17 | 1.035e-17 | 5.867e-17 | 2.123e-18 |
| 15.0 | 1.092e-17 | 9.652e-18 | 6.306e-18 | 1.904e-17 | 7.752e-19 |

### D. BER Comparison Table
| SNR (dB) | LI+DnCNN+AxialAttention Inferred | LI+DnCNN+CrossAttention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.483831 | 0.486087 | 0.456013 | 0.482047 | 0.454276 |
| -5.0 | 0.458521 | 0.457272 | 0.411688 | 0.449682 | 0.411011 |
| 0.0 | 0.389522 | 0.394616 | 0.327831 | 0.380375 | 0.320705 |
| 5.0 | 0.270630 | 0.275977 | 0.220744 | 0.273889 | 0.207732 |
| 10.0 | 0.149915 | 0.149622 | 0.122571 | 0.160101 | 0.101399 |
| 15.0 | 0.060511 | 0.056614 | 0.056730 | 0.068865 | 0.032343 |

