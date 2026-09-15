# Overall Synthesized Results & Dataset Directory Notes

**Generated Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/syn/syn_1`

This document notes the exact source folders, file paths, visual configurations (labels, colors, markers), and metric performance summary for all datasets included in the comparative plots.

> **Note on Benchmark Averaging:**
> The **LS+LI Benchmark** and **LMMSE Benchmark** curves on the plots represent the **mean metric values averaged across all loaded model datasets/approaches** to provide a unified baseline comparison.

--- 

## 1. Selected Folder Sources & Visual Configurations

| # | Model / Curve Label | Source Synthesized Directory | MAT File Path | Color (RGB) | Marker |
|:---:|:---|:---|:---|:---:|:---:|
| 1 | **LS+Attention+DualDomain** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LS_Attention_DualDomain/LS_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_DualDomain\LS_synthesize\synthesized_results.mat` | `[0.850, 0.325, 0.098]` | `^` |
| 2 | **LS+Attention+DualDomain std** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LS_Attention_DualDomain_standardize/LS_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_DualDomain_standardize\LS_synthesize\synthesized_results.mat` | `[0.466, 0.674, 0.188]` | `v` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Attention+DualDomain | LS+Attention+DualDomain std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -3.47 dB | -3.82 dB | 13.05 dB | -2.72 dB |
| -5.0 | -6.50 dB | -6.68 dB | 8.21 dB | -7.32 dB |
| 0.0 | -9.79 dB | -10.32 dB | 3.29 dB | -11.82 dB |
| 5.0 | -13.94 dB | -13.43 dB | -1.93 dB | -16.34 dB |
| 10.0 | -16.53 dB | -18.05 dB | -6.81 dB | -20.59 dB |
| 15.0 | -20.96 dB | -22.30 dB | -11.89 dB | -25.08 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Attention+DualDomain | LS+Attention+DualDomain std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.3434 | 0.3612 | 0.0065 | 0.1047 |
| -5.0 | 0.5058 | 0.5071 | 0.0222 | 0.4846 |
| 0.0 | 0.7519 | 0.7652 | 0.1238 | 0.8190 |
| 5.0 | 0.8825 | 0.8572 | 0.3249 | 0.9317 |
| 10.0 | 0.9326 | 0.9509 | 0.6076 | 0.9690 |
| 15.0 | 0.9720 | 0.9764 | 0.7996 | 0.9870 |

### C. MMSE Comparison Table
| SNR (dB) | LS+Attention+DualDomain | LS+Attention+DualDomain std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.205e-16 | 1.094e-16 | 5.649e-15 | 1.361e-16 |
| -5.0 | 6.461e-17 | 6.196e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 2.752e-17 | 2.471e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 1.095e-17 | 1.270e-17 | 1.997e-16 | 5.395e-18 |
| 10.0 | 5.474e-18 | 4.151e-18 | 6.144e-17 | 2.211e-18 |
| 15.0 | 2.209e-18 | 1.617e-18 | 2.133e-17 | 8.671e-19 |

### D. BER Comparison Table
| SNR (dB) | LS+Attention+DualDomain | LS+Attention+DualDomain std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.455654 | 0.455197 | 0.482474 | 0.453490 |
| -5.0 | 0.407894 | 0.406665 | 0.449624 | 0.407575 |
| 0.0 | 0.322541 | 0.320988 | 0.380368 | 0.318218 |
| 5.0 | 0.206839 | 0.207605 | 0.269748 | 0.201977 |
| 10.0 | 0.106831 | 0.103555 | 0.159096 | 0.099420 |
| 15.0 | 0.034083 | 0.032395 | 0.067497 | 0.030274 |

