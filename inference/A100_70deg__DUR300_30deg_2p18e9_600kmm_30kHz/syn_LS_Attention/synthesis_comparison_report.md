# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/syn_1`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LS+Transformer Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention` |
| 2 | **LS+Transformer Std Inferred** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz/LS_Attention_standardize` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Transformer Inferred | LS+Transformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -1.53 dB | -2.52 dB | 10.89 dB | -2.12 dB |
| -5.0 | -4.14 dB | -4.15 dB | 5.89 dB | -6.11 dB |
| 0.0 | -5.94 dB | -5.91 dB | 0.90 dB | -9.22 dB |
| 5.0 | -7.10 dB | -7.24 dB | -4.09 dB | -13.32 dB |
| 10.0 | -7.47 dB | -7.88 dB | -9.06 dB | -16.86 dB |
| 15.0 | -7.96 dB | -8.18 dB | -13.95 dB | -20.56 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Transformer Inferred | LS+Transformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1172 | 0.1954 | 0.0078 | 0.0710 |
| -5.0 | 0.3037 | 0.3019 | 0.0480 | 0.4364 |
| 0.0 | 0.4234 | 0.4261 | 0.1818 | 0.6605 |
| 5.0 | 0.5191 | 0.5139 | 0.4265 | 0.8283 |
| 10.0 | 0.5742 | 0.5839 | 0.6758 | 0.9134 |
| 15.0 | 0.6169 | 0.6211 | 0.8415 | 0.9574 |

### C. MSE Comparison Table
| SNR (dB) | LS+Transformer Inferred | LS+Transformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.802e-19 | 1.745e-19 | 3.632e-18 | 1.761e-19 |
| -5.0 | 1.237e-19 | 1.139e-19 | 1.104e-18 | 6.228e-20 |
| 0.0 | 1.785e-19 | 1.794e-19 | 5.979e-19 | 3.815e-20 |
| 5.0 | 6.963e-20 | 6.752e-20 | 1.399e-19 | 9.718e-21 |
| 10.0 | 6.692e-20 | 6.513e-20 | 4.162e-20 | 3.661e-21 |
| 15.0 | 5.794e-20 | 5.426e-20 | 1.678e-20 | 2.277e-21 |

### D. BER Comparison Table
| SNR (dB) | LS+Transformer Inferred | LS+Transformer Std Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.463733 | 0.460657 | 0.478983 | 0.459236 |
| -5.0 | 0.419744 | 0.419737 | 0.442484 | 0.415146 |
| 0.0 | 0.340217 | 0.340343 | 0.363592 | 0.325323 |
| 5.0 | 0.241854 | 0.240697 | 0.249350 | 0.211177 |
| 10.0 | 0.162304 | 0.158318 | 0.134006 | 0.104709 |
| 15.0 | 0.114622 | 0.110726 | 0.051488 | 0.035978 |

