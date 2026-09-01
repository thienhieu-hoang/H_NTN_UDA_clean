# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\syn_5`

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
| -10.0 | -6.22 dB | -9.67 dB | 10.90 dB | -3.90 dB |
| -5.0 | -9.57 dB | -12.94 dB | 5.90 dB | -9.39 dB |
| 0.0 | -11.96 dB | -15.55 dB | 0.90 dB | -13.25 dB |
| 5.0 | -18.33 dB | -19.16 dB | -4.10 dB | -17.00 dB |
| 10.0 | -21.00 dB | -22.73 dB | -9.10 dB | -20.76 dB |
| 15.0 | -24.38 dB | -25.89 dB | -14.09 dB | -24.78 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Attention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.2516 | 0.5340 | 0.0028 | 0.1001 |
| -5.0 | 0.3858 | 0.6966 | 0.0208 | 0.2817 |
| 0.0 | 0.5340 | 0.7663 | 0.0818 | 0.5681 |
| 5.0 | 0.8436 | 0.8586 | 0.2261 | 0.7547 |
| 10.0 | 0.8980 | 0.9298 | 0.4422 | 0.8665 |
| 15.0 | 0.9503 | 0.9631 | 0.6533 | 0.9350 |

### C. MSE Comparison Table
| SNR (dB) | LS+Attention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 5.632e-20 | 2.625e-20 | 3.145e-18 | 1.009e-19 |
| -5.0 | 2.562e-20 | 1.122e-20 | 9.961e-19 | 2.540e-20 |
| 0.0 | 1.449e-20 | 6.736e-21 | 3.122e-19 | 1.047e-20 |
| 5.0 | 3.309e-21 | 2.829e-21 | 9.591e-20 | 4.308e-21 |
| 10.0 | 1.626e-21 | 1.118e-21 | 2.949e-20 | 1.793e-21 |
| 15.0 | 8.316e-22 | 5.997e-22 | 1.006e-20 | 7.799e-22 |

### D. BER Comparison Table
| SNR (dB) | LS+Attention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.449666 | 0.447245 | 0.478966 | 0.449289 |
| -5.0 | 0.400361 | 0.398096 | 0.440834 | 0.403469 |
| 0.0 | 0.309291 | 0.306589 | 0.357410 | 0.307729 |
| 5.0 | 0.189269 | 0.188677 | 0.234593 | 0.189900 |
| 10.0 | 0.079122 | 0.077799 | 0.115154 | 0.079528 |
| 15.0 | 0.015063 | 0.014410 | 0.032287 | 0.014859 |

