# Overall Synthesized Results & Dataset Directory Notes

**Generated Output Directory:**
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/syn/syn_4`

This document notes the exact source folders, file paths, visual configurations (labels, colors, markers), and metric performance summary for all datasets included in the comparative plots.

> **Note on Benchmark Averaging:**
> The **LS+LI Benchmark** and **LMMSE Benchmark** curves on the plots represent the **mean metric values averaged across all loaded model datasets/approaches** to provide a unified baseline comparison.

--- 

## 1. Selected Folder Sources & Visual Configurations

| # | Model / Curve Label | Source Synthesized Directory | MAT File Path | Color (RGB) | Marker |
|:---:|:---|:---|:---|:---:|:---:|
| 1 | **LS+Transformer** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LS_Attention/LS_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention\LS_synthesize\synthesized_results.mat` | `[0.850, 0.325, 0.098]` | `^` |
| 2 | **LS+Transformer (Std)** | `C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/A100_2p18e9_600km_70deg_30kHz/LS_Attention_standardize/LS_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_standardize\LS_synthesize\synthesized_results.mat` | `[0.466, 0.674, 0.188]` | `v` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LS+Transformer | LS+Transformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -2.83 dB | -3.89 dB | 13.05 dB | -2.72 dB |
| -5.0 | -6.56 dB | -6.48 dB | 8.21 dB | -7.32 dB |
| 0.0 | -9.89 dB | -9.98 dB | 3.29 dB | -11.82 dB |
| 5.0 | -13.93 dB | -13.99 dB | -1.93 dB | -16.34 dB |
| 10.0 | -17.43 dB | -18.10 dB | -6.81 dB | -20.59 dB |
| 15.0 | -21.30 dB | -22.58 dB | -11.89 dB | -25.08 dB |

### B. SSIM Comparison Table
| SNR (dB) | LS+Transformer | LS+Transformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.2337 | 0.4087 | 0.0065 | 0.1047 |
| -5.0 | 0.5159 | 0.5187 | 0.0222 | 0.4846 |
| 0.0 | 0.7511 | 0.7621 | 0.1238 | 0.8190 |
| 5.0 | 0.8808 | 0.8859 | 0.3249 | 0.9317 |
| 10.0 | 0.9480 | 0.9502 | 0.6076 | 0.9690 |
| 15.0 | 0.9727 | 0.9781 | 0.7996 | 0.9870 |

### C. MMSE Comparison Table
| SNR (dB) | LS+Transformer | LS+Transformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.266e-16 | 1.138e-16 | 5.649e-15 | 1.361e-16 |
| -5.0 | 6.211e-17 | 6.445e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 2.812e-17 | 2.703e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 1.125e-17 | 1.118e-17 | 1.997e-16 | 5.395e-18 |
| 10.0 | 4.825e-18 | 4.111e-18 | 6.144e-17 | 2.211e-18 |
| 15.0 | 2.028e-18 | 1.509e-18 | 2.133e-17 | 8.671e-19 |

### D. BER Comparison Table
| SNR (dB) | LS+Transformer | LS+Transformer (Std) | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.457575 | 0.455115 | 0.482474 | 0.453490 |
| -5.0 | 0.407323 | 0.407236 | 0.449624 | 0.407575 |
| 0.0 | 0.322052 | 0.322063 | 0.380368 | 0.318218 |
| 5.0 | 0.206900 | 0.206832 | 0.269748 | 0.201977 |
| 10.0 | 0.104527 | 0.103228 | 0.159096 | 0.099420 |
| 15.0 | 0.033376 | 0.031868 | 0.067497 | 0.030274 |

