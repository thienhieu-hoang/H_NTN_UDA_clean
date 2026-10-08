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
| 1 | **LI+DnCNN+rms** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LI_DnCNN_rms/LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LI_DnCNN_rms\LI_synthesize\synthesized_results.mat` | `[0.850, 0.325, 0.098]` | `^` |
| 2 | **LS+Transformer+rms** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LS_Attention_rms/LS_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_rms\LS_synthesize\synthesized_results.mat` | `[0.466, 0.674, 0.188]` | `v` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+DnCNN+rms | LS+Transformer+rms | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -4.01 dB | -3.96 dB | 13.05 dB | -2.72 dB |
| -5.0 | -6.59 dB | -7.12 dB | 8.21 dB | -7.32 dB |
| 0.0 | -9.64 dB | -11.23 dB | 3.29 dB | -11.82 dB |
| 5.0 | -13.41 dB | -15.02 dB | -1.93 dB | -16.34 dB |
| 10.0 | -17.20 dB | -18.94 dB | -6.81 dB | -20.59 dB |
| 15.0 | -21.21 dB | -21.95 dB | -11.89 dB | -25.08 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI+DnCNN+rms | LS+Transformer+rms | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1846 | 0.0850 | 0.0065 | 0.1047 |
| -5.0 | 0.4352 | 0.5336 | 0.0222 | 0.4846 |
| 0.0 | 0.7012 | 0.8028 | 0.1238 | 0.8190 |
| 5.0 | 0.8382 | 0.8979 | 0.3249 | 0.9317 |
| 10.0 | 0.9300 | 0.9569 | 0.6076 | 0.9690 |
| 15.0 | 0.9662 | 0.9783 | 0.7996 | 0.9870 |

### C. MMSE Comparison Table
| SNR (dB) | LI+DnCNN+rms | LS+Transformer+rms | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 9.432e-17 | 8.749e-17 | 5.649e-15 | 1.361e-16 |
| -5.0 | 5.764e-17 | 4.707e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 2.838e-17 | 1.917e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 1.268e-17 | 8.201e-18 | 1.997e-16 | 5.395e-18 |
| 10.0 | 5.215e-18 | 3.239e-18 | 6.144e-17 | 2.211e-18 |
| 15.0 | 2.311e-18 | 1.547e-18 | 2.133e-17 | 8.671e-19 |

### D. BER Comparison Table
| SNR (dB) | LI+DnCNN+rms | LS+Transformer+rms | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.455905 | 0.456624 | 0.482474 | 0.453490 |
| -5.0 | 0.408298 | 0.406131 | 0.449624 | 0.407575 |
| 0.0 | 0.322700 | 0.319713 | 0.380368 | 0.318218 |
| 5.0 | 0.207301 | 0.205238 | 0.269748 | 0.201977 |
| 10.0 | 0.105133 | 0.101758 | 0.159096 | 0.099420 |
| 15.0 | 0.034078 | 0.032826 | 0.067497 | 0.030274 |

