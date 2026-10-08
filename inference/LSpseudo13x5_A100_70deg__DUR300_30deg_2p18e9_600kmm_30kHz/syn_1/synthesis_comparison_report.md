# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x5_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/syn_1`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LS+Transformer RMS inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/LSpseudo13x5_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_rms` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Transformer RMS inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.75 dB | 10.89 dB | -2.12 dB |
| -5.0 | 0.73 dB | 5.89 dB | -6.11 dB |
| 0.0 | -1.37 dB | 0.90 dB | -9.22 dB |
| 5.0 | -2.17 dB | -4.09 dB | -13.32 dB |
| 10.0 | -5.28 dB | -9.06 dB | -16.86 dB |
| 15.0 | -7.06 dB | -13.95 dB | -20.56 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Transformer RMS inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0353 | 0.0078 | 0.0710 |
| -5.0 | 0.1902 | 0.0480 | 0.4364 |
| 0.0 | 0.3174 | 0.1818 | 0.6605 |
| 5.0 | 0.4123 | 0.4265 | 0.8283 |
| 10.0 | 0.5615 | 0.6758 | 0.9134 |
| 15.0 | 0.6331 | 0.8415 | 0.9574 |

### C. MSE Comparison Table
| SNR (dB) | LS+Transformer RMS inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|
| -10.0 | 4.238e-19 | 3.632e-18 | 1.761e-19 |
| -5.0 | 3.623e-19 | 1.104e-18 | 6.228e-20 |
| 0.0 | 4.412e-19 | 5.979e-19 | 3.815e-20 |
| 5.0 | 1.880e-19 | 1.399e-19 | 9.718e-21 |
| 10.0 | 1.073e-19 | 4.162e-20 | 3.661e-21 |
| 15.0 | 7.109e-20 | 1.678e-20 | 2.277e-21 |

### D. BER Comparison Table
| SNR (dB) | LS+Transformer RMS inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.471666 | 0.478983 | 0.459236 |
| -5.0 | 0.433055 | 0.442484 | 0.415146 |
| 0.0 | 0.362957 | 0.363592 | 0.325323 |
| 5.0 | 0.290434 | 0.249350 | 0.211177 |
| 10.0 | 0.193602 | 0.134006 | 0.104709 |
| 15.0 | 0.128144 | 0.051488 | 0.035978 |

