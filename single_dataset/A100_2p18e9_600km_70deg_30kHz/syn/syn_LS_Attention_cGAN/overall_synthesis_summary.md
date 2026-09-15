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
| 1 | **LS+Attention+cGAN** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LS_Attention_cGAN/LS_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_cGAN\LS_synthesize\synthesized_results.mat` | `[0.850, 0.325, 0.098]` | `^` |
| 2 | **LS+Attention+cGAN std** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LS_Attention_cGAN_standardize/LS_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_cGAN_standardize\LS_synthesize\synthesized_results.mat` | `[0.466, 0.674, 0.188]` | `v` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Attention+cGAN | LS+Attention+cGAN std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -3.34 dB | -3.92 dB | 13.05 dB | -2.72 dB |
| -5.0 | -6.08 dB | -6.58 dB | 8.21 dB | -7.32 dB |
| 0.0 | -9.57 dB | -9.97 dB | 3.29 dB | -11.82 dB |
| 5.0 | -13.49 dB | -14.29 dB | -1.93 dB | -16.34 dB |
| 10.0 | -16.68 dB | -17.62 dB | -6.81 dB | -20.59 dB |
| 15.0 | -20.56 dB | -21.81 dB | -11.89 dB | -25.08 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Attention+cGAN | LS+Attention+cGAN std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.2979 | 0.4322 | 0.0065 | 0.1047 |
| -5.0 | 0.4970 | 0.5273 | 0.0222 | 0.4846 |
| 0.0 | 0.7592 | 0.7676 | 0.1238 | 0.8190 |
| 5.0 | 0.8670 | 0.8861 | 0.3249 | 0.9317 |
| 10.0 | 0.9400 | 0.9431 | 0.6076 | 0.9690 |
| 15.0 | 0.9680 | 0.9739 | 0.7996 | 0.9870 |

### C. MMSE Comparison Table
| SNR (dB) | LS+Attention+cGAN | LS+Attention+cGAN std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.142e-16 | 1.040e-16 | 5.649e-15 | 1.361e-16 |
| -5.0 | 6.803e-17 | 5.979e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 3.040e-17 | 2.703e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 1.226e-17 | 1.029e-17 | 1.997e-16 | 5.395e-18 |
| 10.0 | 5.516e-18 | 4.504e-18 | 6.144e-17 | 2.211e-18 |
| 15.0 | 2.331e-18 | 1.732e-18 | 2.133e-17 | 8.671e-19 |

### D. BER Comparison Table
| SNR (dB) | LS+Attention+cGAN | LS+Attention+cGAN std | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.456315 | 0.455298 | 0.482474 | 0.453490 |
| -5.0 | 0.408020 | 0.406738 | 0.449624 | 0.407575 |
| 0.0 | 0.323354 | 0.321898 | 0.380368 | 0.318218 |
| 5.0 | 0.208529 | 0.206136 | 0.269748 | 0.201977 |
| 10.0 | 0.106363 | 0.104120 | 0.159096 | 0.099420 |
| 15.0 | 0.034493 | 0.033005 | 0.067497 | 0.030274 |

