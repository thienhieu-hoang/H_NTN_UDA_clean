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
| 1 | **LI+DnCNN** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LI_DnCNN/LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LI_DnCNN\LI_synthesize\synthesized_results.mat` | `[0.850, 0.325, 0.098]` | `^` |
| 2 | **LI+DnCNN (Std)** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LI_DnCNN_standardize/LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LI_DnCNN_standardize\LI_synthesize\synthesized_results.mat` | `[0.466, 0.674, 0.188]` | `v` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+DnCNN | LI+DnCNN (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -3.36 dB | -3.90 dB | 13.05 dB | -2.72 dB |
| -5.0 | -6.03 dB | -6.69 dB | 8.21 dB | -7.32 dB |
| 0.0 | -9.41 dB | -9.74 dB | 3.29 dB | -11.82 dB |
| 5.0 | -13.70 dB | -13.94 dB | -1.93 dB | -16.34 dB |
| 10.0 | -17.34 dB | -17.46 dB | -6.81 dB | -20.59 dB |
| 15.0 | -21.38 dB | -21.54 dB | -11.89 dB | -25.08 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+DnCNN | LI+DnCNN (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.2347 | 0.2708 | 0.0065 | 0.1047 |
| -5.0 | 0.4180 | 0.4674 | 0.0222 | 0.4846 |
| 0.0 | 0.6953 | 0.7046 | 0.1238 | 0.8190 |
| 5.0 | 0.8542 | 0.8640 | 0.3249 | 0.9317 |
| 10.0 | 0.9336 | 0.9349 | 0.6076 | 0.9690 |
| 15.0 | 0.9698 | 0.9712 | 0.7996 | 0.9870 |

### C. MMSE Comparison Table
| SNR (dB) | LI+DnCNN | LI+DnCNN (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.138e-16 | 1.002e-16 | 5.649e-15 | 1.361e-16 |
| -5.0 | 6.691e-17 | 5.726e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 2.982e-17 | 2.742e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 1.162e-17 | 1.102e-17 | 1.997e-16 | 5.395e-18 |
| 10.0 | 4.892e-18 | 4.826e-18 | 6.144e-17 | 2.211e-18 |
| 15.0 | 2.142e-18 | 2.055e-18 | 2.133e-17 | 8.671e-19 |

### D. BER Comparison Table
| SNR (dB) | LI+DnCNN | LI+DnCNN (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.456640 | 0.455489 | 0.482474 | 0.453490 |
| -5.0 | 0.408668 | 0.406894 | 0.449624 | 0.407575 |
| 0.0 | 0.322937 | 0.321869 | 0.380368 | 0.318218 |
| 5.0 | 0.206858 | 0.206627 | 0.269748 | 0.201977 |
| 10.0 | 0.104753 | 0.104373 | 0.159096 | 0.099420 |
| 15.0 | 0.033673 | 0.033629 | 0.067497 | 0.030274 |

