# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x3_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/syn_2`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LS+Transformer stdFDA minmaxPreProcess inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x3_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_stdFDA_minmaxReTrain` |
| 2 | **LS+Transformer stdFDA RMSPreProcess inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x3_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_stdFDA_rmsReTrain` |
| 3 | **LS+Transformer stdFDA stdPreProcess inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x3_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_stdFDA_stdReTrain` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Transformer stdFDA minmaxPreProcess inferred | LS+Transformer stdFDA RMSPreProcess inferred | LS+Transformer stdFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -2.13 dB | -2.73 dB | -2.54 dB | 10.89 dB | -2.12 dB |
| -5.0 | -4.26 dB | -4.50 dB | -4.11 dB | 5.89 dB | -6.11 dB |
| 0.0 | -6.07 dB | -6.43 dB | -6.23 dB | 0.90 dB | -9.22 dB |
| 5.0 | -7.59 dB | -7.83 dB | -7.67 dB | -4.09 dB | -13.32 dB |
| 10.0 | -8.27 dB | -8.78 dB | -8.52 dB | -9.06 dB | -16.86 dB |
| 15.0 | -8.97 dB | -9.02 dB | -9.80 dB | -13.95 dB | -20.56 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Transformer stdFDA minmaxPreProcess inferred | LS+Transformer stdFDA RMSPreProcess inferred | LS+Transformer stdFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1643 | 0.1390 | 0.1841 | 0.0078 | 0.0710 |
| -5.0 | 0.3096 | 0.3131 | 0.3083 | 0.0480 | 0.4364 |
| 0.0 | 0.4427 | 0.4577 | 0.4409 | 0.1818 | 0.6605 |
| 5.0 | 0.5531 | 0.5767 | 0.5490 | 0.4265 | 0.8283 |
| 10.0 | 0.6184 | 0.6527 | 0.6220 | 0.6758 | 0.9134 |
| 15.0 | 0.6744 | 0.6831 | 0.7103 | 0.8415 | 0.9574 |

### C. MSE Comparison Table
| SNR (dB) | LS+Transformer stdFDA minmaxPreProcess inferred | LS+Transformer stdFDA RMSPreProcess inferred | LS+Transformer stdFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.634e-19 | 1.275e-19 | 1.500e-19 | 3.632e-18 | 1.761e-19 |
| -5.0 | 1.081e-19 | 1.099e-19 | 1.071e-19 | 1.104e-18 | 6.228e-20 |
| 0.0 | 1.556e-19 | 1.389e-19 | 1.505e-19 | 5.979e-19 | 3.815e-20 |
| 5.0 | 5.825e-20 | 5.252e-20 | 5.902e-20 | 1.399e-19 | 9.718e-21 |
| 10.0 | 5.972e-20 | 5.928e-20 | 5.936e-20 | 4.162e-20 | 3.661e-21 |
| 15.0 | 4.345e-20 | 4.702e-20 | 3.547e-20 | 1.678e-20 | 2.277e-21 |

### D. BER Comparison Table
| SNR (dB) | LS+Transformer stdFDA minmaxPreProcess inferred | LS+Transformer stdFDA RMSPreProcess inferred | LS+Transformer stdFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.461765 | 0.461965 | 0.460415 | 0.478983 | 0.459236 |
| -5.0 | 0.419056 | 0.418114 | 0.419619 | 0.442484 | 0.415146 |
| 0.0 | 0.339103 | 0.335893 | 0.337937 | 0.363592 | 0.325323 |
| 5.0 | 0.238237 | 0.234785 | 0.237525 | 0.249350 | 0.211177 |
| 10.0 | 0.154466 | 0.147356 | 0.152538 | 0.134006 | 0.104709 |
| 15.0 | 0.101966 | 0.097088 | 0.091686 | 0.051488 | 0.035978 |

