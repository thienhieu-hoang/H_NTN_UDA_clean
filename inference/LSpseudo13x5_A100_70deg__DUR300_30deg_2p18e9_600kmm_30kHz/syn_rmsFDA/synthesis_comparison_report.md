# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x5_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/syn_4`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LS+Transformer rmsFDA minmaxPreProcess inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x5_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_rmsFDA_minmaxReTrain` |
| 2 | **LS+Transformer rmsFDA RMSPreProcess inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x5_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_rmsFDA_rmsReTrain` |
| 3 | **LS+Transformer rmsFDA stdPreProcess inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x5_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_rmsFDA_stdReTrain` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Transformer rmsFDA minmaxPreProcess inferred | LS+Transformer rmsFDA RMSPreProcess inferred | LS+Transformer rmsFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -2.09 dB | -2.37 dB | -2.39 dB | 10.89 dB | -2.12 dB |
| -5.0 | -4.30 dB | -4.64 dB | -4.35 dB | 5.89 dB | -6.11 dB |
| 0.0 | -5.91 dB | -6.46 dB | -6.19 dB | 0.90 dB | -9.22 dB |
| 5.0 | -7.54 dB | -7.66 dB | -7.65 dB | -4.09 dB | -13.32 dB |
| 10.0 | -8.46 dB | -8.73 dB | -8.71 dB | -9.06 dB | -16.86 dB |
| 15.0 | -8.88 dB | -9.07 dB | -9.49 dB | -13.95 dB | -20.56 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Transformer rmsFDA minmaxPreProcess inferred | LS+Transformer rmsFDA RMSPreProcess inferred | LS+Transformer rmsFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1651 | 0.0530 | 0.1900 | 0.0078 | 0.0710 |
| -5.0 | 0.3101 | 0.3266 | 0.3160 | 0.0480 | 0.4364 |
| 0.0 | 0.4470 | 0.4623 | 0.4474 | 0.1818 | 0.6605 |
| 5.0 | 0.5555 | 0.5780 | 0.5528 | 0.4265 | 0.8283 |
| 10.0 | 0.6318 | 0.6520 | 0.6355 | 0.6758 | 0.9134 |
| 15.0 | 0.6828 | 0.6777 | 0.6971 | 0.8415 | 0.9574 |

### C. MSE Comparison Table
| SNR (dB) | LS+Transformer rmsFDA minmaxPreProcess inferred | LS+Transformer rmsFDA RMSPreProcess inferred | LS+Transformer rmsFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.561e-19 | 1.504e-19 | 1.541e-19 | 3.632e-18 | 1.761e-19 |
| -5.0 | 1.091e-19 | 1.033e-19 | 1.100e-19 | 1.104e-18 | 6.228e-20 |
| 0.0 | 1.654e-19 | 1.528e-19 | 1.406e-19 | 5.979e-19 | 3.815e-20 |
| 5.0 | 5.698e-20 | 5.473e-20 | 5.987e-20 | 1.399e-19 | 9.718e-21 |
| 10.0 | 5.855e-20 | 5.701e-20 | 5.741e-20 | 4.162e-20 | 3.661e-21 |
| 15.0 | 4.512e-20 | 4.312e-20 | 3.755e-20 | 1.678e-20 | 2.277e-21 |

### D. BER Comparison Table
| SNR (dB) | LS+Transformer rmsFDA minmaxPreProcess inferred | LS+Transformer rmsFDA RMSPreProcess inferred | LS+Transformer rmsFDA stdPreProcess inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.461612 | 0.463484 | 0.460729 | 0.478983 | 0.459236 |
| -5.0 | 0.418936 | 0.417423 | 0.418554 | 0.442484 | 0.415146 |
| 0.0 | 0.339746 | 0.336002 | 0.338370 | 0.363592 | 0.325323 |
| 5.0 | 0.238584 | 0.235234 | 0.237651 | 0.249350 | 0.211177 |
| 10.0 | 0.152392 | 0.147928 | 0.150012 | 0.134006 | 0.104709 |
| 15.0 | 0.100966 | 0.099328 | 0.094234 | 0.051488 | 0.035978 |

