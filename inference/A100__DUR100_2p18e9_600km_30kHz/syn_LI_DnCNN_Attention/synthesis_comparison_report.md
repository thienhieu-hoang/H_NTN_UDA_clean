# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/syn_2`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LI+DnCNN+Transformer Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LI_DnCNN_Attention` |
| 2 | **LI+DnCNN+Transformer(Std) Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LI_DnCNN_Attention_standardize` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+DnCNN+Transformer Inferred | LI+DnCNN+Transformer(Std) Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 13.97 dB | 23.65 dB | 10.90 dB | -3.90 dB |
| -5.0 | 8.38 dB | 18.77 dB | 5.90 dB | -9.39 dB |
| 0.0 | 3.23 dB | 13.83 dB | 0.90 dB | -13.25 dB |
| 5.0 | -2.60 dB | 8.01 dB | -4.10 dB | -17.00 dB |
| 10.0 | -8.10 dB | 2.80 dB | -9.10 dB | -20.76 dB |
| 15.0 | -13.96 dB | -3.52 dB | -14.09 dB | -24.78 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+DnCNN+Transformer Inferred | LI+DnCNN+Transformer(Std) Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.0034 | -0.0001 | 0.0028 | 0.1001 |
| -5.0 | 0.0230 | 0.0001 | 0.0208 | 0.2817 |
| 0.0 | 0.0763 | -0.0003 | 0.0818 | 0.5681 |
| 5.0 | 0.2438 | 0.0153 | 0.2261 | 0.7547 |
| 10.0 | 0.4302 | 0.1049 | 0.4422 | 0.8665 |
| 15.0 | 0.6494 | 0.2833 | 0.6533 | 0.9350 |

### C. MSE Comparison Table
| SNR (dB) | LI+DnCNN+Transformer Inferred | LI+DnCNN+Transformer(Std) Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 6.380e-18 | 5.965e-17 | 3.145e-18 | 1.009e-19 |
| -5.0 | 1.771e-18 | 1.941e-17 | 9.961e-19 | 2.540e-20 |
| 0.0 | 5.322e-19 | 6.023e-18 | 3.122e-19 | 1.047e-20 |
| 5.0 | 1.356e-19 | 1.584e-18 | 9.591e-20 | 4.308e-21 |
| 10.0 | 3.889e-20 | 4.805e-19 | 2.949e-20 | 1.793e-21 |
| 15.0 | 1.085e-20 | 1.225e-19 | 1.006e-20 | 7.799e-22 |

### D. BER Comparison Table
| SNR (dB) | LI+DnCNN+Transformer Inferred | LI+DnCNN+Transformer(Std) Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.481419 | 0.496759 | 0.478966 | 0.449289 |
| -5.0 | 0.446602 | 0.489034 | 0.440834 | 0.403469 |
| 0.0 | 0.366795 | 0.451557 | 0.357410 | 0.307729 |
| 5.0 | 0.247689 | 0.382013 | 0.234593 | 0.189900 |
| 10.0 | 0.117596 | 0.248866 | 0.115154 | 0.079528 |
| 15.0 | 0.031922 | 0.119175 | 0.032287 | 0.014859 |

