# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/syn_1`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LI+cGAN Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LI_cGAN` |
| 2 | **LI+cGAN Std Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LI_cGAN_standardize` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+cGAN Inferred | LI+cGAN Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 12.44 dB | 14.04 dB | 10.90 dB | -3.90 dB |
| -5.0 | 8.40 dB | 12.20 dB | 5.90 dB | -9.39 dB |
| 0.0 | 4.32 dB | 8.89 dB | 0.90 dB | -13.25 dB |
| 5.0 | 1.19 dB | 6.55 dB | -4.10 dB | -17.00 dB |
| 10.0 | -2.25 dB | 3.43 dB | -9.10 dB | -20.76 dB |
| 15.0 | -5.24 dB | 0.65 dB | -14.09 dB | -24.78 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+cGAN Inferred | LI+cGAN Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.0042 | 0.0028 | 0.0028 | 0.1001 |
| -5.0 | 0.0263 | 0.0170 | 0.0208 | 0.2817 |
| 0.0 | 0.0845 | 0.0454 | 0.0818 | 0.5681 |
| 5.0 | 0.2052 | 0.1027 | 0.2261 | 0.7547 |
| 10.0 | 0.3660 | 0.2098 | 0.4422 | 0.8665 |
| 15.0 | 0.5047 | 0.3256 | 0.6533 | 0.9350 |

### C. MSE Comparison Table
| SNR (dB) | LI+cGAN Inferred | LI+cGAN Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 4.530e-18 | 6.541e-18 | 3.145e-18 | 1.009e-19 |
| -5.0 | 1.749e-18 | 4.068e-18 | 9.961e-19 | 2.540e-20 |
| 0.0 | 6.701e-19 | 1.856e-18 | 3.122e-19 | 1.047e-20 |
| 5.0 | 3.172e-19 | 1.052e-18 | 9.591e-20 | 4.308e-21 |
| 10.0 | 1.325e-19 | 4.883e-19 | 2.949e-20 | 1.793e-21 |
| 15.0 | 7.039e-20 | 2.719e-19 | 1.006e-20 | 7.799e-22 |

### D. BER Comparison Table
| SNR (dB) | LI+cGAN Inferred | LI+cGAN Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.476371 | 0.483893 | 0.478966 | 0.449289 |
| -5.0 | 0.448702 | 0.462838 | 0.440834 | 0.403469 |
| 0.0 | 0.395521 | 0.422868 | 0.357410 | 0.307729 |
| 5.0 | 0.305161 | 0.371796 | 0.234593 | 0.189900 |
| 10.0 | 0.216738 | 0.311818 | 0.115154 | 0.079528 |
| 15.0 | 0.143477 | 0.261148 | 0.032287 | 0.014859 |

