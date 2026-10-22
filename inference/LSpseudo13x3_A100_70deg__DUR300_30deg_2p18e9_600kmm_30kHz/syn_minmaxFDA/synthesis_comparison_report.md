# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x3_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/syn_1`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LS+Transformer minmaxFDA minmaxPreProcess inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x3_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_minmaxFDA_minmaxReTrain` |
| 2 | **LS+Transformer minmaxFDA RMSPreProcess inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x3_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_minmaxFDA_rmsReTrain` |
| 3 | **LS+Transformer minmaxFDA stdPreProcess inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x3_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_minmaxFDA_stdReTrain` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Transformer minmaxFDA minmaxPreProcess inferred | LS+Transformer minmaxFDA RMSPreProcess inferred | LS+Transformer minmaxFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -1.84 dB | -2.81 dB | -2.35 dB | 10.89 dB | -2.12 dB |
| -5.0 | -4.21 dB | -4.61 dB | -4.24 dB | 5.89 dB | -6.11 dB |
| 0.0 | -5.93 dB | -6.34 dB | -6.24 dB | 0.90 dB | -9.22 dB |
| 5.0 | -7.80 dB | -7.49 dB | -7.76 dB | -4.09 dB | -13.32 dB |
| 10.0 | -8.46 dB | -8.41 dB | -8.75 dB | -9.06 dB | -16.86 dB |
| 15.0 | -9.32 dB | -9.50 dB | -9.60 dB | -13.95 dB | -20.56 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Transformer minmaxFDA minmaxPreProcess inferred | LS+Transformer minmaxFDA RMSPreProcess inferred | LS+Transformer minmaxFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1538 | 0.1661 | 0.1867 | 0.0078 | 0.0710 |
| -5.0 | 0.3095 | 0.3251 | 0.3100 | 0.0480 | 0.4364 |
| 0.0 | 0.4458 | 0.4608 | 0.4500 | 0.1818 | 0.6605 |
| 5.0 | 0.5662 | 0.5641 | 0.5557 | 0.4265 | 0.8283 |
| 10.0 | 0.6263 | 0.6334 | 0.6379 | 0.6758 | 0.9134 |
| 15.0 | 0.7003 | 0.6998 | 0.7050 | 0.8415 | 0.9574 |

### C. MSE Comparison Table
| SNR (dB) | LS+Transformer minmaxFDA minmaxPreProcess inferred | LS+Transformer minmaxFDA RMSPreProcess inferred | LS+Transformer minmaxFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.854e-19 | 1.389e-19 | 1.499e-19 | 3.632e-18 | 1.761e-19 |
| -5.0 | 1.068e-19 | 1.008e-19 | 1.033e-19 | 1.104e-18 | 6.228e-20 |
| 0.0 | 1.606e-19 | 1.557e-19 | 1.433e-19 | 5.979e-19 | 3.815e-20 |
| 5.0 | 5.611e-20 | 5.886e-20 | 5.664e-20 | 1.399e-19 | 9.718e-21 |
| 10.0 | 5.813e-20 | 5.995e-20 | 5.784e-20 | 4.162e-20 | 3.661e-21 |
| 15.0 | 3.952e-20 | 3.735e-20 | 3.673e-20 | 1.678e-20 | 2.277e-21 |

### D. BER Comparison Table
| SNR (dB) | LS+Transformer minmaxFDA minmaxPreProcess inferred | LS+Transformer minmaxFDA RMSPreProcess inferred | LS+Transformer minmaxFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.462263 | 0.461118 | 0.460885 | 0.478983 | 0.459236 |
| -5.0 | 0.419236 | 0.417318 | 0.418931 | 0.442484 | 0.415146 |
| 0.0 | 0.339696 | 0.335783 | 0.337923 | 0.363592 | 0.325323 |
| 5.0 | 0.236753 | 0.236759 | 0.236934 | 0.249350 | 0.211177 |
| 10.0 | 0.152870 | 0.150405 | 0.149964 | 0.134006 | 0.104709 |
| 15.0 | 0.096934 | 0.093651 | 0.093287 | 0.051488 | 0.035978 |

