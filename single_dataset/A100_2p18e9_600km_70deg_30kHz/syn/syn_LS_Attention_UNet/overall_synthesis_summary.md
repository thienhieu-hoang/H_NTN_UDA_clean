# Overall Synthesized Results & Dataset Directory Notes

**Generated Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/syn/syn_3`

This document notes the exact source folders, file paths, visual configurations (labels, colors, markers), and metric performance summary for all datasets included in the comparative plots.

> **Note on Benchmark Averaging:**
> The **LS+LI Benchmark** and **LMMSE Benchmark** curves on the plots represent the **mean metric values averaged across all loaded model datasets/approaches** to provide a unified baseline comparison.

--- 

## 1. Selected Folder Sources & Visual Configurations

| # | Model / Curve Label | Source Synthesized Directory | MAT File Path | Color (RGB) | Marker |
|:---:|:---|:---|:---|:---:|:---:|
| 1 | **LS+Attention+UNet** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LS_Attention_UNetRefine/LS_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_UNetRefine\LS_synthesize\synthesized_results.mat` | `[0.850, 0.325, 0.098]` | `^` |
| 2 | **LS+Attention+UNet std** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LS_Attention_UNetRefine_standardize/LS_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_UNetRefine_standardize\LS_synthesize\synthesized_results.mat` | `[0.466, 0.674, 0.188]` | `v` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Attention+UNet | LS+Attention+UNet std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -3.41 dB | -4.62 dB | 13.05 dB | -2.72 dB |
| -5.0 | -6.92 dB | -7.16 dB | 8.21 dB | -7.32 dB |
| 0.0 | -10.06 dB | -10.48 dB | 3.29 dB | -11.82 dB |
| 5.0 | -13.60 dB | -14.94 dB | -1.93 dB | -16.34 dB |
| 10.0 | -17.32 dB | -18.68 dB | -6.81 dB | -20.59 dB |
| 15.0 | -21.27 dB | -22.07 dB | -11.89 dB | -25.08 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Attention+UNet | LS+Attention+UNet std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.3254 | 0.3241 | 0.0065 | 0.1047 |
| -5.0 | 0.5156 | 0.4950 | 0.0222 | 0.4846 |
| 0.0 | 0.7562 | 0.7752 | 0.1238 | 0.8190 |
| 5.0 | 0.8679 | 0.9019 | 0.3249 | 0.9317 |
| 10.0 | 0.9469 | 0.9511 | 0.6076 | 0.9690 |
| 15.0 | 0.9729 | 0.9766 | 0.7996 | 0.9870 |

### C. MMSE Comparison Table
| SNR (dB) | LS+Attention+UNet | LS+Attention+UNet std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.121e-16 | 8.355e-17 | 5.649e-15 | 1.361e-16 |
| -5.0 | 5.408e-17 | 4.921e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 2.530e-17 | 2.237e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 1.104e-17 | 8.663e-18 | 1.997e-16 | 5.395e-18 |
| 10.0 | 4.643e-18 | 3.492e-18 | 6.144e-17 | 2.211e-18 |
| 15.0 | 2.076e-18 | 1.626e-18 | 2.133e-17 | 8.671e-19 |

### D. BER Comparison Table
| SNR (dB) | LS+Attention+UNet | LS+Attention+UNet std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.455953 | 0.453520 | 0.482474 | 0.453490 |
| -5.0 | 0.406168 | 0.406147 | 0.449624 | 0.407575 |
| 0.0 | 0.321672 | 0.320752 | 0.380368 | 0.318218 |
| 5.0 | 0.206609 | 0.204983 | 0.269748 | 0.201977 |
| 10.0 | 0.105239 | 0.102158 | 0.159096 | 0.099420 |
| 15.0 | 0.033664 | 0.032737 | 0.067497 | 0.030274 |

