# Overall Synthesized Results & Dataset Directory Notes

**Generated Output Directory:**
`C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_syn`

This document notes the exact source folders, file paths, visual configurations (labels, colors, markers), and metric performance summary for all datasets included in the comparative plots.

> **Note on Benchmark Averaging:**
> The **LS+LI Benchmark** and **LMMSE Benchmark** curves on the plots represent the **mean metric values averaged across all loaded model datasets/approaches** to provide a unified baseline comparison.

--- 

## 1. Selected Folder Sources & Visual Configurations

| # | Model / Curve Label | Source Synthesized Directory | MAT File Path | Color (RGB) | Marker |
|:---:|:---|:---|:---|:---:|:---:|
| 1 | **LI+DnCNN+Attention** | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_LI_DnCNN_Attention\LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_LI_DnCNN_Attention\LI_synthesize\synthesized_results.mat` | `[0.000, 0.125, 0.376]` | `o` |
| 2 | **LI+DnCNN** | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_LI_DnCNN\LI_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_LI_DnCNN\LI_synthesize\synthesized_results.mat` | `[0.000, 0.439, 0.753]` | `s` |
| 3 | **LS+Attention** | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_LS_Attention_standardize\LS_synthesize` | `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_LS_Attention_standardize\LS_synthesize\synthesized_results.mat` | `[0.753, 0.000, 0.000]` | `^` |

--- 

## 2. Comparative Metric Summaries Across SNRs

### A. NMSE (dB) Comparison Table
| SNR (dB) | LI+DnCNN+Attention | LI+DnCNN | LS+Attention | Avg LS+LI Bench | Avg LMMSE Bench |\n|:---:|:---:|:---:|:---:|:---:|:---:|\n| -10.0 | -4.77 dB | -5.04 dB | -5.76 dB | 10.73 dB | -3.78 dB |\n| -5.0 | -8.08 dB | -8.18 dB | -8.69 dB | 5.73 dB | -9.27 dB |\n| 0.0 | -11.60 dB | -11.57 dB | -11.86 dB | 0.74 dB | -13.42 dB |\n| 5.0 | -14.97 dB | -15.20 dB | -15.20 dB | -4.26 dB | -17.23 dB |\n| 10.0 | -18.85 dB | -18.91 dB | -18.63 dB | -9.26 dB | -20.80 dB |\n| 15.0 | -22.68 dB | -22.52 dB | -22.02 dB | -14.27 dB | -24.92 dB |\n
### B. SSIM Comparison Table
| SNR (dB) | LI+DnCNN+Attention | LI+DnCNN | LS+Attention | Avg LS+LI Bench | Avg LMMSE Bench |\n|:---:|:---:|:---:|:---:|:---:|:---:|\n| -10.0 | 0.0480 | 0.0511 | 0.1785 | 0.0026 | 0.0897 |\n| -5.0 | 0.1566 | 0.1681 | 0.3636 | 0.0229 | 0.2984 |\n| 0.0 | 0.3949 | 0.3880 | 0.5414 | 0.0826 | 0.5559 |\n| 5.0 | 0.6187 | 0.6253 | 0.6824 | 0.2211 | 0.7456 |\n| 10.0 | 0.7900 | 0.7965 | 0.8153 | 0.4289 | 0.8626 |\n| 15.0 | 0.9088 | 0.9060 | 0.9030 | 0.6602 | 0.9380 |\n
### C. MMSE Comparison Table
| SNR (dB) | LI+DnCNN+Attention | LI+DnCNN | LS+Attention | Avg LS+LI Bench | Avg LMMSE Bench |\n|:---:|:---:|:---:|:---:|:---:|:---:|\n| -10.0 | 8.686e-20 | 8.242e-20 | 7.242e-20 | 3.008e-18 | 1.021e-19 |\n| -5.0 | 3.317e-20 | 3.283e-20 | 2.904e-20 | 7.947e-19 | 2.528e-20 |\n| 0.0 | 1.559e-20 | 1.596e-20 | 1.465e-20 | 2.643e-19 | 1.016e-20 |\n| 5.0 | 7.450e-21 | 7.072e-21 | 7.851e-21 | 8.998e-20 | 4.377e-21 |\n| 10.0 | 2.675e-21 | 2.635e-21 | 2.751e-21 | 2.656e-20 | 1.619e-21 |\n| 15.0 | 1.313e-21 | 1.382e-21 | 1.555e-21 | 9.644e-21 | 7.557e-22 |\n
### D. BER Comparison Table
| SNR (dB) | LI+DnCNN+Attention | LI+DnCNN | LS+Attention | Avg LS+LI Bench | Avg LMMSE Bench |\n|:---:|:---:|:---:|:---:|:---:|:---:|\n| -10.0 | 0.452231 | 0.451247 | 0.450246 | 0.479784 | 0.449531 |\n| -5.0 | 0.400823 | 0.400549 | 0.398855 | 0.439853 | 0.400477 |\n| 0.0 | 0.305554 | 0.305550 | 0.305558 | 0.354951 | 0.302614 |\n| 5.0 | 0.187436 | 0.187297 | 0.187954 | 0.230741 | 0.184133 |\n| 10.0 | 0.077438 | 0.077610 | 0.078191 | 0.111106 | 0.074767 |\n| 15.0 | 0.013652 | 0.013628 | 0.014454 | 0.029431 | 0.012488 |\n
