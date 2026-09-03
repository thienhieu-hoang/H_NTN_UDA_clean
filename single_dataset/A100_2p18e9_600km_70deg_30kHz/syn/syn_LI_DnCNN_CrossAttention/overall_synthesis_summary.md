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
| 1 | **LI+DnCNN+CrossTransformer** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LI_DnCNN_CrossAttention/LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LI_DnCNN_CrossAttention\LI_synthesize\synthesized_results.mat` | `[0.850, 0.325, 0.098]` | `^` |
| 2 | **LI+DnCNN+CrossTransformer (Std)** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LI_DnCNN_CrossAttention_standardize/LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LI_DnCNN_CrossAttention_standardize\LI_synthesize\synthesized_results.mat` | `[0.466, 0.674, 0.188]` | `v` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+DnCNN+CrossTransformer | LI+DnCNN+CrossTransformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -3.01 dB | -3.47 dB | 13.05 dB | -2.72 dB |
| -5.0 | -6.08 dB | -6.34 dB | 8.21 dB | -7.32 dB |
| 0.0 | -9.37 dB | -9.75 dB | 3.29 dB | -11.82 dB |
| 5.0 | -13.56 dB | -13.96 dB | -1.93 dB | -16.34 dB |
| 10.0 | -17.26 dB | -17.41 dB | -6.81 dB | -20.59 dB |
| 15.0 | -21.29 dB | -21.60 dB | -11.89 dB | -25.08 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+DnCNN+CrossTransformer | LI+DnCNN+CrossTransformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.2186 | 0.2466 | 0.0065 | 0.1047 |
| -5.0 | 0.4171 | 0.4434 | 0.0222 | 0.4846 |
| 0.0 | 0.6933 | 0.7048 | 0.1238 | 0.8190 |
| 5.0 | 0.8503 | 0.8636 | 0.3249 | 0.9317 |
| 10.0 | 0.9333 | 0.9376 | 0.6076 | 0.9690 |
| 15.0 | 0.9680 | 0.9701 | 0.7996 | 0.9870 |

### C. MMSE Comparison Table
| SNR (dB) | LI+DnCNN+CrossTransformer | LI+DnCNN+CrossTransformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.257e-16 | 1.147e-16 | 5.649e-15 | 1.361e-16 |
| -5.0 | 6.586e-17 | 6.197e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 3.028e-17 | 2.746e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 1.199e-17 | 1.089e-17 | 1.997e-16 | 5.395e-18 |
| 10.0 | 5.039e-18 | 4.837e-18 | 6.144e-17 | 2.211e-18 |
| 15.0 | 2.200e-18 | 2.048e-18 | 2.133e-17 | 8.671e-19 |

### D. BER Comparison Table
| SNR (dB) | LI+DnCNN+CrossTransformer | LI+DnCNN+CrossTransformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.457261 | 0.456864 | 0.482474 | 0.453490 |
| -5.0 | 0.408649 | 0.407431 | 0.449624 | 0.407575 |
| 0.0 | 0.323017 | 0.321908 | 0.380368 | 0.318218 |
| 5.0 | 0.207243 | 0.206751 | 0.269748 | 0.201977 |
| 10.0 | 0.105058 | 0.104847 | 0.159096 | 0.099420 |
| 15.0 | 0.033821 | 0.033584 | 0.067497 | 0.030274 |

