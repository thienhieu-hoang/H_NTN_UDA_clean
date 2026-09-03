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
| 1 | **LI+cGAN** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LI_cGAN/LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LI_cGAN\LI_synthesize\synthesized_results.mat` | `[0.850, 0.325, 0.098]` | `^` |
| 2 | **LI+cGAN (Std)** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LI_cGAN_standardize/LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LI_cGAN_standardize\LI_synthesize\synthesized_results.mat` | `[0.466, 0.674, 0.188]` | `v` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+cGAN | LI+cGAN (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -0.17 dB | -4.31 dB | 13.05 dB | -2.72 dB |
| -5.0 | -4.23 dB | -7.22 dB | 8.21 dB | -7.32 dB |
| 0.0 | -7.78 dB | -10.64 dB | 3.29 dB | -11.82 dB |
| 5.0 | -11.38 dB | -14.92 dB | -1.93 dB | -16.34 dB |
| 10.0 | -14.35 dB | -18.60 dB | -6.81 dB | -20.59 dB |
| 15.0 | -18.03 dB | -22.64 dB | -11.89 dB | -25.08 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+cGAN | LI+cGAN (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.2546 | 0.3690 | 0.0065 | 0.1047 |
| -5.0 | 0.4766 | 0.5645 | 0.0222 | 0.4846 |
| 0.0 | 0.7078 | 0.7680 | 0.1238 | 0.8190 |
| 5.0 | 0.8579 | 0.8936 | 0.3249 | 0.9317 |
| 10.0 | 0.9293 | 0.9532 | 0.6076 | 0.9690 |
| 15.0 | 0.9646 | 0.9785 | 0.7996 | 0.9870 |

### C. MMSE Comparison Table
| SNR (dB) | LI+cGAN | LI+cGAN (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 2.478e-16 | 9.051e-17 | 5.649e-15 | 1.361e-16 |
| -5.0 | 9.645e-17 | 5.152e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 4.642e-17 | 2.263e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 1.921e-17 | 8.729e-18 | 1.997e-16 | 5.395e-18 |
| 10.0 | 9.319e-18 | 3.727e-18 | 6.144e-17 | 2.211e-18 |
| 15.0 | 4.017e-18 | 1.535e-18 | 2.133e-17 | 8.671e-19 |

### D. BER Comparison Table
| SNR (dB) | LI+cGAN | LI+cGAN (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.463370 | 0.454550 | 0.482474 | 0.453490 |
| -5.0 | 0.413214 | 0.405565 | 0.449624 | 0.407575 |
| 0.0 | 0.329262 | 0.320378 | 0.380368 | 0.318218 |
| 5.0 | 0.214248 | 0.204840 | 0.269748 | 0.201977 |
| 10.0 | 0.115742 | 0.102157 | 0.159096 | 0.099420 |
| 15.0 | 0.041866 | 0.032127 | 0.067497 | 0.030274 |

