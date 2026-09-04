# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/syn_4`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LI+DnCNN+CrossTransformer Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LI_DnCNN_CrossAttention` |
| 2 | **LI+DnCNN+CrossTransformer(Std) Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LI_DnCNN_CrossAttention_standardize` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+DnCNN+CrossTransformer Inferred | LI+DnCNN+CrossTransformer(Std) Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 12.90 dB | 23.61 dB | 10.90 dB | -3.90 dB |
| -5.0 | 8.49 dB | 18.29 dB | 5.90 dB | -9.39 dB |
| 0.0 | 2.51 dB | 13.77 dB | 0.90 dB | -13.25 dB |
| 5.0 | -1.94 dB | 8.31 dB | -4.10 dB | -17.00 dB |
| 10.0 | -7.76 dB | 2.08 dB | -9.10 dB | -20.76 dB |
| 15.0 | -13.66 dB | -3.72 dB | -14.09 dB | -24.78 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+DnCNN+CrossTransformer Inferred | LI+DnCNN+CrossTransformer(Std) Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.0050 | -0.0001 | 0.0028 | 0.1001 |
| -5.0 | 0.0211 | 0.0002 | 0.0208 | 0.2817 |
| 0.0 | 0.1039 | -0.0009 | 0.0818 | 0.5681 |
| 5.0 | 0.2308 | 0.0120 | 0.2261 | 0.7547 |
| 10.0 | 0.4340 | 0.1118 | 0.4422 | 0.8665 |
| 15.0 | 0.6441 | 0.2902 | 0.6533 | 0.9350 |

### C. MSE Comparison Table
| SNR (dB) | LI+DnCNN+CrossTransformer Inferred | LI+DnCNN+CrossTransformer(Std) Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 5.030e-18 | 5.923e-17 | 3.145e-18 | 1.009e-19 |
| -5.0 | 1.830e-18 | 1.745e-17 | 9.961e-19 | 2.540e-20 |
| 0.0 | 4.508e-19 | 5.979e-18 | 3.122e-19 | 1.047e-20 |
| 5.0 | 1.615e-19 | 1.723e-18 | 9.591e-20 | 4.308e-21 |
| 10.0 | 4.280e-20 | 4.076e-19 | 2.949e-20 | 1.793e-21 |
| 15.0 | 1.165e-20 | 1.188e-19 | 1.006e-20 | 7.799e-22 |

### D. BER Comparison Table
| SNR (dB) | LI+DnCNN+CrossTransformer Inferred | LI+DnCNN+CrossTransformer(Std) Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.480438 | 0.497160 | 0.478966 | 0.449289 |
| -5.0 | 0.445317 | 0.482566 | 0.440834 | 0.403469 |
| 0.0 | 0.362414 | 0.454199 | 0.357410 | 0.307729 |
| 5.0 | 0.249160 | 0.373017 | 0.234593 | 0.189900 |
| 10.0 | 0.118258 | 0.247168 | 0.115154 | 0.079528 |
| 15.0 | 0.033318 | 0.116008 | 0.032287 | 0.014859 |

