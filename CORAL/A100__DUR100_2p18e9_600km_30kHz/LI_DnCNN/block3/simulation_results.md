# CORAL UDA Channel Equalization & Domain Performance Summary

**Batch Directory:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\CORAL\A100__DUR100_2p18e9_600km_30kHz\LI_DnCNN\block3`  
**Model Evaluation:** `LI+DnCNN CORAL (block3)`  

## Multi-Domain BER Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.459441 | 0.482506 | 0.479818 | 0.449321 |
| -5.0 | 0.412855 | 0.470108 | 0.441466 | 0.403807 |
| 0.0 | 0.326954 | 0.432377 | 0.357242 | 0.306450 |
| 5.0 | 0.213785 | 0.379717 | 0.233443 | 0.188221 |
| 10.0 | 0.112244 | 0.301362 | 0.113611 | 0.078185 |
| 15.0 | 0.043575 | 0.234533 | 0.030859 | 0.013600 |

## Multi-Domain NMSE (dB) Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -2.35 dB | 13.51 dB | 10.73 dB | -3.78 dB |
| -5.0 | -5.48 dB | 10.97 dB | 5.73 dB | -9.27 dB |
| 0.0 | -9.06 dB | 7.03 dB | 0.74 dB | -13.42 dB |
| 5.0 | -12.61 dB | 3.22 dB | -4.26 dB | -17.23 dB |
| 10.0 | -15.32 dB | -0.14 dB | -9.26 dB | -20.80 dB |
| 15.0 | -16.58 dB | -2.56 dB | -14.27 dB | -24.92 dB |

## Multi-Domain SSIM Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.2116 | -0.0036 | 0.0026 | 0.0897 |
| -5.0 | 0.3974 | -0.0050 | 0.0229 | 0.2984 |
| 0.0 | 0.6842 | 0.0015 | 0.0826 | 0.5559 |
| 5.0 | 0.8262 | -0.0383 | 0.2211 | 0.7456 |
| 10.0 | 0.9056 | 0.0222 | 0.4289 | 0.8626 |
| 15.0 | 0.9249 | 0.1215 | 0.6602 | 0.9380 |
