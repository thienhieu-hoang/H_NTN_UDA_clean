# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\syn_1`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LS+Attention Inferred** | `C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\LS_Attention` |
| 2 | **LS+Attention Std Inferred** | `C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\LS_Attention_standardize` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Attention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -2.75 dB | -3.83 dB | 13.15 dB | -2.69 dB |
| -5.0 | -5.43 dB | -6.32 dB | 8.21 dB | -7.36 dB |
| 0.0 | -7.32 dB | -8.64 dB | 3.25 dB | -11.91 dB |
| 5.0 | -10.37 dB | -10.67 dB | -1.84 dB | -16.38 dB |
| 10.0 | -11.82 dB | -12.63 dB | -6.79 dB | -20.72 dB |
| 15.0 | -12.72 dB | -14.26 dB | -11.82 dB | -25.13 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Attention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1522 | 0.2315 | 0.0047 | 0.1009 |
| -5.0 | 0.3526 | 0.4338 | 0.0248 | 0.5012 |
| 0.0 | 0.5370 | 0.6584 | 0.1196 | 0.8124 |
| 5.0 | 0.7578 | 0.7756 | 0.3379 | 0.9283 |
| 10.0 | 0.8302 | 0.8584 | 0.6121 | 0.9714 |
| 15.0 | 0.8656 | 0.9031 | 0.8087 | 0.9885 |

### C. MSE Comparison Table
| SNR (dB) | LS+Attention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.299e-16 | 9.978e-17 | 5.798e-15 | 1.406e-16 |
| -5.0 | 6.998e-17 | 5.651e-17 | 1.924e-15 | 4.385e-17 |
| 0.0 | 3.975e-17 | 2.986e-17 | 5.973e-16 | 1.397e-17 |
| 5.0 | 1.902e-17 | 1.775e-17 | 1.938e-16 | 5.567e-18 |
| 10.0 | 1.216e-17 | 1.035e-17 | 5.867e-17 | 2.123e-18 |
| 15.0 | 8.666e-18 | 6.306e-18 | 1.904e-17 | 7.752e-19 |

### D. BER Comparison Table
| SNR (dB) | LS+Attention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.458510 | 0.456013 | 0.482047 | 0.454276 |
| -5.0 | 0.414284 | 0.411688 | 0.449682 | 0.411011 |
| 0.0 | 0.331894 | 0.327831 | 0.380375 | 0.320705 |
| 5.0 | 0.221784 | 0.220744 | 0.273889 | 0.207732 |
| 10.0 | 0.126389 | 0.122571 | 0.160101 | 0.101399 |
| 15.0 | 0.065822 | 0.056730 | 0.068865 | 0.032343 |

