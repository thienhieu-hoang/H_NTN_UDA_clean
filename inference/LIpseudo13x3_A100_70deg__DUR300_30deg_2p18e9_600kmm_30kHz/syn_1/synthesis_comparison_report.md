# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LIpseudo13x3_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/syn_1`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LI+DnCNN RMS inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LIpseudo13x3_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LI_DnCNN_rms` |
| 2 | **LS+Transformer RMS inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LIpseudo13x3_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_rms` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+DnCNN RMS inferred | LS+Transformer RMS inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 19.17 dB | 1.49 dB | 10.89 dB | -2.12 dB |
| -5.0 | 13.75 dB | -0.02 dB | 5.89 dB | -6.11 dB |
| 0.0 | 6.78 dB | -1.98 dB | 0.90 dB | -9.22 dB |
| 5.0 | -0.63 dB | -5.15 dB | -4.09 dB | -13.32 dB |
| 10.0 | -9.11 dB | -8.31 dB | -9.06 dB | -16.86 dB |
| 15.0 | -16.83 dB | -10.91 dB | -13.95 dB | -20.56 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+DnCNN RMS inferred | LS+Transformer RMS inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -0.0005 | 0.1029 | 0.0078 | 0.0710 |
| -5.0 | -0.0015 | 0.2301 | 0.0480 | 0.4364 |
| 0.0 | 0.0349 | 0.4227 | 0.1818 | 0.6605 |
| 5.0 | 0.2885 | 0.6192 | 0.4265 | 0.8283 |
| 10.0 | 0.7035 | 0.7625 | 0.6758 | 0.9134 |
| 15.0 | 0.9069 | 0.8471 | 0.8415 | 0.9574 |

### C. MSE Comparison Table
| SNR (dB) | LI+DnCNN RMS inferred | LS+Transformer RMS inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 2.617e-17 | 4.651e-19 | 3.632e-18 | 1.761e-19 |
| -5.0 | 6.035e-18 | 3.065e-19 | 1.104e-18 | 6.228e-20 |
| 0.0 | 1.817e-18 | 3.875e-19 | 5.979e-19 | 3.815e-20 |
| 5.0 | 2.826e-19 | 1.022e-19 | 1.399e-19 | 9.718e-21 |
| 10.0 | 4.394e-20 | 4.766e-20 | 4.162e-20 | 3.661e-21 |
| 15.0 | 9.096e-21 | 3.034e-20 | 1.678e-20 | 2.277e-21 |

### D. BER Comparison Table
| SNR (dB) | LI+DnCNN RMS inferred | LS+Transformer RMS inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.497037 | 0.466537 | 0.478983 | 0.459236 |
| -5.0 | 0.483769 | 0.426597 | 0.442484 | 0.415146 |
| 0.0 | 0.418989 | 0.352597 | 0.363592 | 0.325323 |
| 5.0 | 0.280179 | 0.256760 | 0.249350 | 0.211177 |
| 10.0 | 0.135078 | 0.159715 | 0.134006 | 0.104709 |
| 15.0 | 0.044529 | 0.082788 | 0.051488 | 0.035978 |

