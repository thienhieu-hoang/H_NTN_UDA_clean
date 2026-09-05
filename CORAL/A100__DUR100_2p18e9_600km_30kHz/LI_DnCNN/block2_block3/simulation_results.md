# CORAL UDA Channel Equalization & Domain Performance Summary

**Batch Directory:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\CORAL\A100__DUR100_2p18e9_600km_30kHz\LI_DnCNN\block2_block3`  
**Model Evaluation:** `LI+DnCNN CORAL (block2_block3)`  

## Multi-Domain BER Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.459796 | 0.484857 | 0.479818 | 0.449321 |
| -5.0 | 0.413127 | 0.467242 | 0.441466 | 0.403807 |
| 0.0 | 0.328442 | 0.432998 | 0.357242 | 0.306450 |
| 5.0 | 0.212888 | 0.376938 | 0.233443 | 0.188221 |
| 10.0 | 0.116580 | 0.302701 | 0.113611 | 0.078185 |
| 15.0 | 0.039352 | 0.231030 | 0.030859 | 0.013600 |

## Multi-Domain NMSE (dB) Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -1.98 dB | 13.51 dB | 10.73 dB | -3.78 dB |
| -5.0 | -5.55 dB | 10.72 dB | 5.73 dB | -9.27 dB |
| 0.0 | -8.67 dB | 7.06 dB | 0.74 dB | -13.42 dB |
| 5.0 | -13.04 dB | 3.15 dB | -4.26 dB | -17.23 dB |
| 10.0 | -14.00 dB | -0.10 dB | -9.26 dB | -20.80 dB |
| 15.0 | -18.91 dB | -2.66 dB | -14.27 dB | -24.92 dB |

## Multi-Domain SSIM Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1703 | -0.0063 | 0.0026 | 0.0897 |
| -5.0 | 0.3811 | -0.0101 | 0.0229 | 0.2984 |
| 0.0 | 0.6793 | 0.0094 | 0.0826 | 0.5559 |
| 5.0 | 0.8377 | -0.0114 | 0.2211 | 0.7456 |
| 10.0 | 0.8727 | 0.0485 | 0.4289 | 0.8626 |
| 15.0 | 0.9515 | 0.0024 | 0.6602 | 0.9380 |
