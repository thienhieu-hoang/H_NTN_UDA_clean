# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/syn_3`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LI+DnCNN+AxialTransformer Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LI_DnCNN_AxialAttention` |
| 2 | **LI+DnCNN+AxialTransformer(Std) Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LI_DnCNN_AxialAttention_standardize` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+DnCNN+AxialTransformer Inferred | LI+DnCNN+AxialTransformer(Std) Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 12.67 dB | 24.75 dB | 10.90 dB | -3.90 dB |
| -5.0 | 7.45 dB | 18.09 dB | 5.90 dB | -9.39 dB |
| 0.0 | 3.43 dB | 13.41 dB | 0.90 dB | -13.25 dB |
| 5.0 | -2.84 dB | 7.39 dB | -4.10 dB | -17.00 dB |
| 10.0 | -8.72 dB | 1.62 dB | -9.10 dB | -20.76 dB |
| 15.0 | -13.97 dB | -4.00 dB | -14.09 dB | -24.78 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+DnCNN+AxialTransformer Inferred | LI+DnCNN+AxialTransformer(Std) Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.0047 | -0.0001 | 0.0028 | 0.1001 |
| -5.0 | 0.0263 | -0.0002 | 0.0208 | 0.2817 |
| 0.0 | 0.0900 | -0.0011 | 0.0818 | 0.5681 |
| 5.0 | 0.2503 | 0.0149 | 0.2261 | 0.7547 |
| 10.0 | 0.4493 | 0.1191 | 0.4422 | 0.8665 |
| 15.0 | 0.6637 | 0.3145 | 0.6533 | 0.9350 |

### C. MSE Comparison Table
| SNR (dB) | LI+DnCNN+AxialTransformer Inferred | LI+DnCNN+AxialTransformer(Std) Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 4.746e-18 | 7.735e-17 | 3.145e-18 | 1.009e-19 |
| -5.0 | 1.417e-18 | 1.666e-17 | 9.961e-19 | 2.540e-20 |
| 0.0 | 5.501e-19 | 5.564e-18 | 3.122e-19 | 1.047e-20 |
| 5.0 | 1.315e-19 | 1.351e-18 | 9.591e-20 | 4.308e-21 |
| 10.0 | 3.371e-20 | 3.622e-19 | 2.949e-20 | 1.793e-21 |
| 15.0 | 1.099e-20 | 1.121e-19 | 1.006e-20 | 7.799e-22 |

### D. BER Comparison Table
| SNR (dB) | LI+DnCNN+AxialTransformer Inferred | LI+DnCNN+AxialTransformer(Std) Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.480875 | 0.495472 | 0.478966 | 0.449289 |
| -5.0 | 0.446573 | 0.484736 | 0.440834 | 0.403469 |
| 0.0 | 0.371394 | 0.455021 | 0.357410 | 0.307729 |
| 5.0 | 0.245172 | 0.367949 | 0.234593 | 0.189900 |
| 10.0 | 0.115642 | 0.244158 | 0.115154 | 0.079528 |
| 15.0 | 0.032403 | 0.117344 | 0.032287 | 0.014859 |

