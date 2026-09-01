# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\A100__DUR100_2p18e9_600km_30kHz\\syn_1`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LS+Attention Inferred** | `C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\A100__DUR100_2p18e9_600km_30kHz\\LS_Attention` |
| 2 | **LS+Attention Std Inferred** | `C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\A100__DUR100_2p18e9_600km_30kHz\\LS_Attention_standardize` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Attention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -3.64 dB | -4.17 dB | 10.90 dB | -3.90 dB |
| -5.0 | -6.65 dB | -6.69 dB | 5.90 dB | -9.39 dB |
| 0.0 | -9.17 dB | -9.92 dB | 0.90 dB | -13.25 dB |
| 5.0 | -12.28 dB | -12.92 dB | -4.10 dB | -17.00 dB |
| 10.0 | -14.55 dB | -14.85 dB | -9.10 dB | -20.76 dB |
| 15.0 | -15.98 dB | -16.57 dB | -14.09 dB | -24.78 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Attention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1390 | 0.1508 | 0.0028 | 0.1001 |
| -5.0 | 0.2590 | 0.2703 | 0.0208 | 0.2817 |
| 0.0 | 0.4008 | 0.4039 | 0.0818 | 0.5681 |
| 5.0 | 0.5470 | 0.5527 | 0.2261 | 0.7547 |
| 10.0 | 0.6598 | 0.6719 | 0.4422 | 0.8665 |
| 15.0 | 0.7339 | 0.7443 | 0.6533 | 0.9350 |

### C. MSE Comparison Table
| SNR (dB) | LS+Attention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.098e-19 | 9.565e-20 | 3.145e-18 | 1.009e-19 |
| -5.0 | 5.356e-20 | 5.306e-20 | 9.961e-19 | 2.540e-20 |
| 0.0 | 2.954e-20 | 2.470e-20 | 3.122e-19 | 1.047e-20 |
| 5.0 | 1.318e-20 | 1.156e-20 | 9.591e-20 | 4.308e-21 |
| 10.0 | 7.656e-21 | 7.000e-21 | 2.949e-20 | 1.793e-21 |
| 15.0 | 6.016e-21 | 5.239e-21 | 1.006e-20 | 7.799e-22 |

### D. BER Comparison Table
| SNR (dB) | LS+Attention Inferred | LS+Attention Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.452466 | 0.451750 | 0.478966 | 0.449289 |
| -5.0 | 0.404551 | 0.404362 | 0.440834 | 0.403469 |
| 0.0 | 0.315560 | 0.313618 | 0.357410 | 0.307729 |
| 5.0 | 0.199515 | 0.197282 | 0.234593 | 0.189900 |
| 10.0 | 0.093327 | 0.092094 | 0.115154 | 0.079528 |
| 15.0 | 0.029287 | 0.027291 | 0.032287 | 0.014859 |

