# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/syn_1`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LI+DnCNN RMS inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LI_DnCNN_rms` |
| 2 | **LS+Transformer RMS inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_rms` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+DnCNN RMS inferred | LS+Transformer RMS inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 19.15 dB | 0.92 dB | 10.89 dB | -2.12 dB |
| -5.0 | 13.30 dB | 2.42 dB | 5.89 dB | -6.11 dB |
| 0.0 | 6.91 dB | -0.07 dB | 0.90 dB | -9.22 dB |
| 5.0 | -1.05 dB | -2.86 dB | -4.09 dB | -13.32 dB |
| 10.0 | -8.60 dB | -4.67 dB | -9.06 dB | -16.86 dB |
| 15.0 | -15.59 dB | -6.03 dB | -13.95 dB | -20.56 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+DnCNN RMS inferred | LS+Transformer RMS inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -0.0008 | 0.0485 | 0.0078 | 0.0710 |
| -5.0 | -0.0045 | 0.1298 | 0.0480 | 0.4364 |
| 0.0 | 0.0068 | 0.2649 | 0.1818 | 0.6605 |
| 5.0 | 0.2311 | 0.3715 | 0.4265 | 0.8283 |
| 10.0 | 0.6398 | 0.5096 | 0.6758 | 0.9134 |
| 15.0 | 0.8835 | 0.5730 | 0.8415 | 0.9574 |

### C. MSE Comparison Table
| SNR (dB) | LI+DnCNN RMS inferred | LS+Transformer RMS inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 2.618e-17 | 4.050e-19 | 3.632e-18 | 1.761e-19 |
| -5.0 | 5.375e-18 | 5.877e-19 | 1.104e-18 | 6.228e-20 |
| 0.0 | 2.010e-18 | 6.494e-19 | 5.979e-19 | 3.815e-20 |
| 5.0 | 2.645e-19 | 1.757e-19 | 1.399e-19 | 9.718e-21 |
| 10.0 | 4.731e-20 | 1.171e-19 | 4.162e-20 | 3.661e-21 |
| 15.0 | 1.088e-20 | 8.913e-20 | 1.678e-20 | 2.277e-21 |

### D. BER Comparison Table
| SNR (dB) | LI+DnCNN RMS inferred | LS+Transformer RMS inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.499665 | 0.472507 | 0.478983 | 0.459236 |
| -5.0 | 0.486627 | 0.439319 | 0.442484 | 0.415146 |
| 0.0 | 0.428303 | 0.374489 | 0.363592 | 0.325323 |
| 5.0 | 0.285040 | 0.284281 | 0.249350 | 0.211177 |
| 10.0 | 0.142225 | 0.201529 | 0.134006 | 0.104709 |
| 15.0 | 0.049666 | 0.141610 | 0.051488 | 0.035978 |

