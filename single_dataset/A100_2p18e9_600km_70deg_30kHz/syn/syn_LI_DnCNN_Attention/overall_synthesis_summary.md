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
| 1 | **LI+DnCNN+Transformer** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LI_DnCNN_Attention/LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LI_DnCNN_Attention\LI_synthesize\synthesized_results.mat` | `[0.850, 0.325, 0.098]` | `^` |
| 2 | **LI+DnCNN+Transformer (Std)** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LI_DnCNN_Attention_standardize/LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LI_DnCNN_Attention_standardize\LI_synthesize\synthesized_results.mat` | `[0.466, 0.674, 0.188]` | `v` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+DnCNN+Transformer | LI+DnCNN+Transformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -3.09 dB | -3.79 dB | 13.05 dB | -2.72 dB |
| -5.0 | -6.11 dB | -6.58 dB | 8.21 dB | -7.32 dB |
| 0.0 | -9.16 dB | -9.66 dB | 3.29 dB | -11.82 dB |
| 5.0 | -13.39 dB | -13.74 dB | -1.93 dB | -16.34 dB |
| 10.0 | -17.21 dB | -17.41 dB | -6.81 dB | -20.59 dB |
| 15.0 | -21.39 dB | -21.63 dB | -11.89 dB | -25.08 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+DnCNN+Transformer | LI+DnCNN+Transformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.2257 | 0.2676 | 0.0065 | 0.1047 |
| -5.0 | 0.4219 | 0.4644 | 0.0222 | 0.4846 |
| 0.0 | 0.6785 | 0.7040 | 0.1238 | 0.8190 |
| 5.0 | 0.8458 | 0.8625 | 0.3249 | 0.9317 |
| 10.0 | 0.9314 | 0.9361 | 0.6076 | 0.9690 |
| 15.0 | 0.9691 | 0.9709 | 0.7996 | 0.9870 |

### C. MMSE Comparison Table
| SNR (dB) | LI+DnCNN+Transformer | LI+DnCNN+Transformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.227e-16 | 1.017e-16 | 5.649e-15 | 1.361e-16 |
| -5.0 | 6.537e-17 | 5.867e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 3.135e-17 | 2.810e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 1.265e-17 | 1.165e-17 | 1.997e-16 | 5.395e-18 |
| 10.0 | 5.034e-18 | 4.775e-18 | 6.144e-17 | 2.211e-18 |
| 15.0 | 2.135e-18 | 2.013e-18 | 2.133e-17 | 8.671e-19 |

### D. BER Comparison Table
| SNR (dB) | LI+DnCNN+Transformer | LI+DnCNN+Transformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.456961 | 0.455683 | 0.482474 | 0.453490 |
| -5.0 | 0.408391 | 0.407151 | 0.449624 | 0.407575 |
| 0.0 | 0.323848 | 0.321988 | 0.380368 | 0.318218 |
| 5.0 | 0.207719 | 0.207406 | 0.269748 | 0.201977 |
| 10.0 | 0.104840 | 0.104563 | 0.159096 | 0.099420 |
| 15.0 | 0.033681 | 0.033410 | 0.067497 | 0.030274 |

