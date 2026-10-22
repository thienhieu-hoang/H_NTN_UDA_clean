# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x5_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/syn_3`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LS+Transformer minmaxFDA minmaxPreProcess inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x5_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_minmaxFDA_minmaxReTrain` |
| 2 | **LS+Transformer minmaxFDA RMSPreProcess inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x5_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_minmaxFDA_rmsReTrain` |
| 3 | **LS+Transformer minmaxFDA stdPreProcess inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x5_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_minmaxFDA_stdReTrain` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Transformer minmaxFDA minmaxPreProcess inferred | LS+Transformer minmaxFDA RMSPreProcess inferred | LS+Transformer minmaxFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -2.06 dB | -2.57 dB | -2.44 dB | 10.89 dB | -2.12 dB |
| -5.0 | -4.17 dB | -4.20 dB | -4.31 dB | 5.89 dB | -6.11 dB |
| 0.0 | -6.10 dB | -6.29 dB | -6.32 dB | 0.90 dB | -9.22 dB |
| 5.0 | -7.72 dB | -7.76 dB | -7.72 dB | -4.09 dB | -13.32 dB |
| 10.0 | -8.54 dB | -8.57 dB | -8.73 dB | -9.06 dB | -16.86 dB |
| 15.0 | -8.88 dB | -9.36 dB | -9.09 dB | -13.95 dB | -20.56 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Transformer minmaxFDA minmaxPreProcess inferred | LS+Transformer minmaxFDA RMSPreProcess inferred | LS+Transformer minmaxFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1712 | 0.1091 | 0.1873 | 0.0078 | 0.0710 |
| -5.0 | 0.3109 | 0.2825 | 0.3168 | 0.0480 | 0.4364 |
| 0.0 | 0.4505 | 0.4649 | 0.4441 | 0.1818 | 0.6605 |
| 5.0 | 0.5661 | 0.5708 | 0.5559 | 0.4265 | 0.8283 |
| 10.0 | 0.6364 | 0.6416 | 0.6386 | 0.6758 | 0.9134 |
| 15.0 | 0.6717 | 0.6988 | 0.6819 | 0.8415 | 0.9574 |

### C. MSE Comparison Table
| SNR (dB) | LS+Transformer minmaxFDA minmaxPreProcess inferred | LS+Transformer minmaxFDA RMSPreProcess inferred | LS+Transformer minmaxFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.719e-19 | 1.384e-19 | 1.483e-19 | 3.632e-18 | 1.761e-19 |
| -5.0 | 1.048e-19 | 1.187e-19 | 1.062e-19 | 1.104e-18 | 6.228e-20 |
| 0.0 | 1.605e-19 | 1.505e-19 | 1.393e-19 | 5.979e-19 | 3.815e-20 |
| 5.0 | 5.755e-20 | 5.410e-20 | 5.828e-20 | 1.399e-19 | 9.718e-21 |
| 10.0 | 5.753e-20 | 5.946e-20 | 5.867e-20 | 4.162e-20 | 3.661e-21 |
| 15.0 | 4.591e-20 | 4.156e-20 | 4.309e-20 | 1.678e-20 | 2.277e-21 |

### D. BER Comparison Table
| SNR (dB) | LS+Transformer minmaxFDA minmaxPreProcess inferred | LS+Transformer minmaxFDA RMSPreProcess inferred | LS+Transformer minmaxFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.461937 | 0.462545 | 0.460826 | 0.478983 | 0.459236 |
| -5.0 | 0.419216 | 0.419335 | 0.418797 | 0.442484 | 0.415146 |
| 0.0 | 0.338776 | 0.335526 | 0.337531 | 0.363592 | 0.325323 |
| 5.0 | 0.237018 | 0.235557 | 0.237292 | 0.249350 | 0.211177 |
| 10.0 | 0.152134 | 0.149193 | 0.150066 | 0.134006 | 0.104709 |
| 15.0 | 0.102881 | 0.093740 | 0.097948 | 0.051488 | 0.035978 |

