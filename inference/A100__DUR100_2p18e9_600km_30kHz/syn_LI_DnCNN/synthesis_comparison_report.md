# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/syn_2`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LI+DnCNN Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LI_DnCNN` |
| 2 | **LI+DnCNN Std Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LI_DnCNN_standardize` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+DnCNN Inferred | LI+DnCNN Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 13.23 dB | 23.66 dB | 10.90 dB | -3.90 dB |
| -5.0 | 7.09 dB | 18.50 dB | 5.90 dB | -9.39 dB |
| 0.0 | 1.96 dB | 14.32 dB | 0.90 dB | -13.25 dB |
| 5.0 | -3.10 dB | 8.15 dB | -4.10 dB | -17.00 dB |
| 10.0 | -7.94 dB | 2.53 dB | -9.10 dB | -20.76 dB |
| 15.0 | -13.69 dB | -3.98 dB | -14.09 dB | -24.78 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+DnCNN Inferred | LI+DnCNN Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -0.0016 | -0.0000 | 0.0028 | 0.1001 |
| -5.0 | 0.0071 | -0.0004 | 0.0208 | 0.2817 |
| 0.0 | 0.0640 | -0.0009 | 0.0818 | 0.5681 |
| 5.0 | 0.2253 | 0.0236 | 0.2261 | 0.7547 |
| 10.0 | 0.4286 | 0.1209 | 0.4422 | 0.8665 |
| 15.0 | 0.6414 | 0.3082 | 0.6533 | 0.9350 |

### C. MSE Comparison Table
| SNR (dB) | LI+DnCNN Inferred | LI+DnCNN Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 5.491e-18 | 5.943e-17 | 3.145e-18 | 1.009e-19 |
| -5.0 | 1.310e-18 | 1.818e-17 | 9.961e-19 | 2.540e-20 |
| 0.0 | 3.958e-19 | 6.790e-18 | 3.122e-19 | 1.047e-20 |
| 5.0 | 1.228e-19 | 1.619e-18 | 9.591e-20 | 4.308e-21 |
| 10.0 | 4.034e-20 | 4.532e-19 | 2.949e-20 | 1.793e-21 |
| 15.0 | 1.201e-20 | 1.112e-19 | 1.006e-20 | 7.799e-22 |

### D. BER Comparison Table
| SNR (dB) | LI+DnCNN Inferred | LI+DnCNN Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.486779 | 0.498345 | 0.478966 | 0.449289 |
| -5.0 | 0.450996 | 0.485532 | 0.440834 | 0.403469 |
| 0.0 | 0.367795 | 0.460556 | 0.357410 | 0.307729 |
| 5.0 | 0.248976 | 0.375007 | 0.234593 | 0.189900 |
| 10.0 | 0.121469 | 0.245789 | 0.115154 | 0.079528 |
| 15.0 | 0.031417 | 0.115971 | 0.032287 | 0.014859 |

