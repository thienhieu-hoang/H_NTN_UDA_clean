# Multi-Model Channel Estimation Synthesis Comparison

**Generated Comparison Output Directory:**
`C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\syn_2`

## 1. Selected Folder Sources & Curve Configurations

| # | Model / Curve Label | Source Directory Path |
|:---:|:---|:---|
| 1 | **LI+cGAN Inferred** | `C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\LI_cGAN` |
| 2 | **LI+DnCNN Inferred** | `C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\LI_DnCNN` |
| 3 | **LI+DnCNN+Attention Inferred** | `C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\LI_DnCNN_Attention` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+cGAN Inferred | LI+DnCNN Inferred | LI+DnCNN+Attention Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 13.82 dB | 10.17 dB | 9.45 dB | 13.15 dB | -2.69 dB |
| -5.0 | 9.37 dB | 5.99 dB | 5.80 dB | 8.21 dB | -7.36 dB |
| 0.0 | 4.52 dB | 1.66 dB | 1.46 dB | 3.25 dB | -11.91 dB |
| 5.0 | 1.08 dB | -3.20 dB | -3.48 dB | -1.84 dB | -16.38 dB |
| 10.0 | -0.68 dB | -8.60 dB | -8.56 dB | -6.79 dB | -20.72 dB |
| 15.0 | -2.53 dB | -14.14 dB | -13.93 dB | -11.82 dB | -25.13 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+cGAN Inferred | LI+DnCNN Inferred | LI+DnCNN+Attention Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.0042 | -0.0036 | -0.0029 | 0.0047 | 0.1009 |
| -5.0 | 0.0241 | -0.0239 | -0.0229 | 0.0248 | 0.5012 |
| 0.0 | 0.1349 | -0.0292 | -0.0226 | 0.1196 | 0.8124 |
| 5.0 | 0.3367 | 0.1982 | 0.2431 | 0.3379 | 0.9283 |
| 10.0 | 0.5193 | 0.6108 | 0.6014 | 0.6121 | 0.9714 |
| 15.0 | 0.6376 | 0.8486 | 0.8429 | 0.8087 | 0.9885 |

### C. MSE Comparison Table
| SNR (dB) | LI+cGAN Inferred | LI+DnCNN Inferred | LI+DnCNN+Attention Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 6.783e-15 | 2.871e-15 | 2.427e-15 | 5.798e-15 | 1.406e-16 |
| -5.0 | 2.503e-15 | 1.119e-15 | 1.066e-15 | 1.924e-15 | 4.385e-17 |
| 0.0 | 7.903e-16 | 4.004e-16 | 3.831e-16 | 5.973e-16 | 1.397e-17 |
| 5.0 | 3.621e-16 | 1.402e-16 | 1.309e-16 | 1.938e-16 | 5.567e-18 |
| 10.0 | 2.197e-16 | 3.727e-17 | 3.772e-17 | 5.867e-17 | 2.123e-18 |
| 15.0 | 1.415e-16 | 9.547e-18 | 1.007e-17 | 1.904e-17 | 7.752e-19 |

### D. BER Comparison Table
| SNR (dB) | LI+cGAN Inferred | LI+DnCNN Inferred | LI+DnCNN+Attention Inferred | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.482858 | 0.484857 | 0.483415 | 0.482047 | 0.454276 |
| -5.0 | 0.456001 | 0.460370 | 0.459658 | 0.449682 | 0.411011 |
| 0.0 | 0.389113 | 0.395165 | 0.392218 | 0.380375 | 0.320705 |
| 5.0 | 0.315611 | 0.275542 | 0.272234 | 0.273889 | 0.207732 |
| 10.0 | 0.264287 | 0.151489 | 0.152397 | 0.160101 | 0.101399 |
| 15.0 | 0.218412 | 0.057792 | 0.058905 | 0.068865 | 0.032343 |

