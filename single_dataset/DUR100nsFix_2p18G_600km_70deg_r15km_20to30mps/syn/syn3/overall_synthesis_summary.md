# Overall Synthesized Results & Dataset Directory Notes

**Generated Output Directory:**
`C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\syn\syn3`

This document notes the exact source folders, file paths, visual configurations (labels, colors, markers), and metric performance summary for all datasets included in the comparative plots.

> **Note on Benchmark Averaging:**
> The **LS+LI Benchmark** and **LMMSE Benchmark** curves on the plots represent the **mean metric values averaged across all loaded model datasets/approaches** to provide a unified baseline comparison.

--- 

## 1. Selected Folder Sources & Visual Configurations

| # | Model / Curve Label | Source Synthesized Directory | MAT File Path | Color (RGB) | Marker |
|:---:|:---|:---|:---|:---:|:---:|
| 1 | **LI + DnCNN + AxialAttention** | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LI_DnCNN_AxialAttention\LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LI_DnCNN_AxialAttention\LI_synthesize\synthesized_results.mat` | `[0.850, 0.325, 0.098]` | `^` |
| 2 | **LI + DnCNN + CrossAttention** | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LI_DnCNN_CrossAttention\LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LI_DnCNN_CrossAttention\LI_synthesize\synthesized_results.mat` | `[0.466, 0.674, 0.188]` | `v` |
| 3 | **LI + DnCNN** | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LI_DnCNN\LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LI_DnCNN\LI_synthesize\synthesized_results.mat` | `[0.494, 0.184, 0.556]` | `d` |
| 4 | **LS + Attention** | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LS_Attention_standardize\LS_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LS_Attention_standardize\LS_synthesize\synthesized_results.mat` | `[0.929, 0.694, 0.125]` | `*` |
| 5 | **LI + cGAN** | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LI_cGAN\LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LI_cGAN\LI_synthesize\synthesized_results.mat` | `[0.301, 0.745, 0.933]` | `p` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI + DnCNN + AxialAttention | LI + DnCNN + CrossAttention | LI + DnCNN | LS + Attention | LI + cGAN | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -5.21 dB | -5.20 dB | -5.04 dB | -5.76 dB | -0.22 dB | 10.73 dB | -3.78 dB |
| -5.0 | -8.28 dB | -8.32 dB | -8.18 dB | -8.69 dB | -4.39 dB | 5.73 dB | -9.27 dB |
| 0.0 | -11.57 dB | -11.63 dB | -11.57 dB | -11.86 dB | -8.34 dB | 0.74 dB | -13.42 dB |
| 5.0 | -15.40 dB | -15.49 dB | -15.20 dB | -15.20 dB | -12.53 dB | -4.26 dB | -17.23 dB |
| 10.0 | -19.21 dB | -18.95 dB | -18.91 dB | -18.63 dB | -16.21 dB | -9.26 dB | -20.80 dB |
| 15.0 | -23.27 dB | -22.85 dB | -22.52 dB | -22.02 dB | -19.56 dB | -14.27 dB | -24.92 dB |

### B. SSIM Comparison Table
| SNR (dB) | LI + DnCNN + AxialAttention | LI + DnCNN + CrossAttention | LI + DnCNN | LS + Attention | LI + cGAN | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.0502 | 0.0380 | 0.0511 | 0.1785 | 0.0679 | 0.0026 | 0.0897 |
| -5.0 | 0.1915 | 0.1689 | 0.1681 | 0.3636 | 0.2252 | 0.0229 | 0.2984 |
| 0.0 | 0.4137 | 0.4069 | 0.3880 | 0.5414 | 0.4354 | 0.0826 | 0.5559 |
| 5.0 | 0.6478 | 0.6269 | 0.6253 | 0.6824 | 0.6335 | 0.2211 | 0.7456 |
| 10.0 | 0.7983 | 0.7935 | 0.7965 | 0.8153 | 0.7889 | 0.4289 | 0.8626 |
| 15.0 | 0.9166 | 0.9122 | 0.9060 | 0.9030 | 0.8950 | 0.6602 | 0.9380 |

### C. MMSE Comparison Table
| SNR (dB) | LI + DnCNN + AxialAttention | LI + DnCNN + CrossAttention | LI + DnCNN | LS + Attention | LI + cGAN | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 7.933e-20 | 7.777e-20 | 8.242e-20 | 7.242e-20 | 2.652e-19 | 3.008e-18 | 1.021e-19 |
| -5.0 | 3.216e-20 | 3.098e-20 | 3.283e-20 | 2.904e-20 | 8.182e-20 | 7.947e-19 | 2.528e-20 |
| 0.0 | 1.609e-20 | 1.551e-20 | 1.596e-20 | 1.465e-20 | 3.245e-20 | 2.643e-19 | 1.016e-20 |
| 5.0 | 6.934e-21 | 6.577e-21 | 7.072e-21 | 7.851e-21 | 1.286e-20 | 8.998e-20 | 4.377e-21 |
| 10.0 | 2.476e-21 | 2.585e-21 | 2.635e-21 | 2.751e-21 | 4.801e-21 | 2.656e-20 | 1.619e-21 |
| 15.0 | 1.177e-21 | 1.261e-21 | 1.382e-21 | 1.555e-21 | 2.513e-21 | 9.644e-21 | 7.557e-22 |

### D. BER Comparison Table
| SNR (dB) | LI + DnCNN + AxialAttention | LI + DnCNN + CrossAttention | LI + DnCNN | LS + Attention | LI + cGAN | Avg LS+LI Bench | Avg LMMSE Bench |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.450256 | 0.451318 | 0.451247 | 0.450246 | 0.461080 | 0.479784 | 0.449531 |
| -5.0 | 0.399719 | 0.399714 | 0.400549 | 0.398855 | 0.408229 | 0.439853 | 0.400477 |
| 0.0 | 0.305497 | 0.305484 | 0.305550 | 0.305558 | 0.313307 | 0.354951 | 0.302614 |
| 5.0 | 0.187429 | 0.186886 | 0.187297 | 0.187954 | 0.193626 | 0.230741 | 0.184133 |
| 10.0 | 0.076907 | 0.077525 | 0.077610 | 0.078191 | 0.084022 | 0.111106 | 0.074767 |
| 15.0 | 0.013369 | 0.013406 | 0.013628 | 0.014454 | 0.017738 | 0.029431 | 0.012488 |

