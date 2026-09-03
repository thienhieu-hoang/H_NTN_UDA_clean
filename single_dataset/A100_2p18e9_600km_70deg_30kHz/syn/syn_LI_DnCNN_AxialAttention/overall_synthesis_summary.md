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
| 1 | **LI+DnCNN+AxialTransformer** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LI_DnCNN_AxialAttention/LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LI_DnCNN_AxialAttention\LI_synthesize\synthesized_results.mat` | `[0.850, 0.325, 0.098]` | `^` |
| 2 | **LI+DnCNN+AxialTransformer (Std)** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LI_DnCNN_AxialAttention_standardize/LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LI_DnCNN_AxialAttention_standardize\LI_synthesize\synthesized_results.mat` | `[0.466, 0.674, 0.188]` | `v` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+DnCNN+AxialTransformer | LI+DnCNN+AxialTransformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -3.41 dB | -3.74 dB | 13.05 dB | -2.72 dB |
| -5.0 | -6.21 dB | -6.61 dB | 8.21 dB | -7.32 dB |
| 0.0 | -9.38 dB | -9.84 dB | 3.29 dB | -11.82 dB |
| 5.0 | -13.54 dB | -14.11 dB | -1.93 dB | -16.34 dB |
| 10.0 | -17.26 dB | -17.58 dB | -6.81 dB | -20.59 dB |
| 15.0 | -21.39 dB | -21.82 dB | -11.89 dB | -25.08 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+DnCNN+AxialTransformer | LI+DnCNN+AxialTransformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.2291 | 0.2656 | 0.0065 | 0.1047 |
| -5.0 | 0.4264 | 0.4695 | 0.0222 | 0.4846 |
| 0.0 | 0.6901 | 0.7170 | 0.1238 | 0.8190 |
| 5.0 | 0.8513 | 0.8675 | 0.3249 | 0.9317 |
| 10.0 | 0.9331 | 0.9389 | 0.6076 | 0.9690 |
| 15.0 | 0.9693 | 0.9722 | 0.7996 | 0.9870 |

### C. MMSE Comparison Table
| SNR (dB) | LI+DnCNN+AxialTransformer | LI+DnCNN+AxialTransformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.131e-16 | 1.041e-16 | 5.649e-15 | 1.361e-16 |
| -5.0 | 6.409e-17 | 5.882e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 2.999e-17 | 2.692e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 1.206e-17 | 1.058e-17 | 1.997e-16 | 5.395e-18 |
| 10.0 | 4.903e-18 | 4.541e-18 | 6.144e-17 | 2.211e-18 |
| 15.0 | 2.105e-18 | 1.899e-18 | 2.133e-17 | 8.671e-19 |

### D. BER Comparison Table
| SNR (dB) | LI+DnCNN+AxialTransformer | LI+DnCNN+AxialTransformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.456237 | 0.455523 | 0.482474 | 0.453490 |
| -5.0 | 0.408102 | 0.407156 | 0.449624 | 0.407575 |
| 0.0 | 0.323004 | 0.321770 | 0.380368 | 0.318218 |
| 5.0 | 0.207077 | 0.206451 | 0.269748 | 0.201977 |
| 10.0 | 0.104649 | 0.104056 | 0.159096 | 0.099420 |
| 15.0 | 0.033641 | 0.033118 | 0.067497 | 0.030274 |

