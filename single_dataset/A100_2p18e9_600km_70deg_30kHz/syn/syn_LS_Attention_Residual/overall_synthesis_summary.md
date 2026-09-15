# Overall Synthesized Results & Dataset Directory Notes

**Generated Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/syn/syn_2`

This document notes the exact source folders, file paths, visual configurations (labels, colors, markers), and metric performance summary for all datasets included in the comparative plots.

> **Note on Benchmark Averaging:**
> The **LS+LI Benchmark** and **LMMSE Benchmark** curves on the plots represent the **mean metric values averaged across all loaded model datasets/approaches** to provide a unified baseline comparison.

--- 

## 1. Selected Folder Sources & Visual Configurations

| # | Model / Curve Label | Source Synthesized Directory | MAT File Path | Color (RGB) | Marker |
|:---:|:---|:---|:---|:---:|:---:|
| 1 | **LS+Attention+Residual** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LS_Attention_ResidualRefine/LS_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_ResidualRefine\LS_synthesize\synthesized_results.mat` | `[0.850, 0.325, 0.098]` | `^` |
| 2 | **LS+Attention+Residual std** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LS_Attention_ResidualRefine_standardize/LS_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_ResidualRefine_standardize\LS_synthesize\synthesized_results.mat` | `[0.466, 0.674, 0.188]` | `v` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Attention+Residual | LS+Attention+Residual std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -3.56 dB | -3.68 dB | 13.05 dB | -2.72 dB |
| -5.0 | -6.37 dB | -6.44 dB | 8.21 dB | -7.32 dB |
| 0.0 | -9.81 dB | -10.10 dB | 3.29 dB | -11.82 dB |
| 5.0 | -13.54 dB | -13.95 dB | -1.93 dB | -16.34 dB |
| 10.0 | -17.67 dB | -17.48 dB | -6.81 dB | -20.59 dB |
| 15.0 | -20.98 dB | -22.28 dB | -11.89 dB | -25.08 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Attention+Residual | LS+Attention+Residual std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.3405 | 0.3558 | 0.0065 | 0.1047 |
| -5.0 | 0.5252 | 0.5310 | 0.0222 | 0.4846 |
| 0.0 | 0.7703 | 0.7749 | 0.1238 | 0.8190 |
| 5.0 | 0.8695 | 0.8751 | 0.3249 | 0.9317 |
| 10.0 | 0.9468 | 0.9433 | 0.6076 | 0.9690 |
| 15.0 | 0.9714 | 0.9772 | 0.7996 | 0.9870 |

### C. MMSE Comparison Table
| SNR (dB) | LS+Attention+Residual | LS+Attention+Residual std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.084e-16 | 1.131e-16 | 5.649e-15 | 1.361e-16 |
| -5.0 | 6.155e-17 | 6.599e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 2.738e-17 | 2.567e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 1.222e-17 | 1.134e-17 | 1.997e-16 | 5.395e-18 |
| 10.0 | 4.514e-18 | 4.622e-18 | 6.144e-17 | 2.211e-18 |
| 15.0 | 2.089e-18 | 1.593e-18 | 2.133e-17 | 8.671e-19 |

### D. BER Comparison Table
| SNR (dB) | LS+Attention+Residual | LS+Attention+Residual std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.455477 | 0.455584 | 0.482474 | 0.453490 |
| -5.0 | 0.407356 | 0.407078 | 0.449624 | 0.407575 |
| 0.0 | 0.322863 | 0.321778 | 0.380368 | 0.318218 |
| 5.0 | 0.207593 | 0.206851 | 0.269748 | 0.201977 |
| 10.0 | 0.104358 | 0.104677 | 0.159096 | 0.099420 |
| 15.0 | 0.033551 | 0.032362 | 0.067497 | 0.030274 |

