# CORAL UDA Channel Equalization & Domain Performance Summary

**Batch Directory:** `C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\CORAL\A100__DUR100_2p18e9_600km_30kHz\LI_DnCNN_pHead\block2_block3`  
**Model Evaluation:** `LI+DnCNN CORAL pHead (block2_block3)`  

## Multi-Domain BER Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.459511 | 0.483141 | 0.479818 | 0.449321 |
| -5.0 | 0.412963 | 0.468276 | 0.441466 | 0.403807 |
| 0.0 | 0.327505 | 0.431681 | 0.357242 | 0.306450 |
| 5.0 | 0.212092 | 0.377058 | 0.233443 | 0.188221 |
| 10.0 | 0.113081 | 0.303953 | 0.113611 | 0.078185 |
| 15.0 | 0.084738 | 0.228947 | 0.030859 | 0.013600 |

## Multi-Domain NMSE (dB) Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -1.84 dB | 13.45 dB | 10.73 dB | -3.78 dB |
| -5.0 | -5.44 dB | 10.77 dB | 5.73 dB | -9.27 dB |
| 0.0 | -8.84 dB | 7.22 dB | 0.74 dB | -13.42 dB |
| 5.0 | -13.43 dB | 3.27 dB | -4.26 dB | -17.23 dB |
| 10.0 | -15.16 dB | -0.12 dB | -9.26 dB | -20.80 dB |
| 15.0 | -7.79 dB | -2.73 dB | -14.27 dB | -24.92 dB |

## Multi-Domain SSIM Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1979 | -0.0014 | 0.0026 | 0.0897 |
| -5.0 | 0.3840 | -0.0075 | 0.0229 | 0.2984 |
| 0.0 | 0.6773 | 0.0036 | 0.0826 | 0.5559 |
| 5.0 | 0.8479 | 0.0058 | 0.2211 | 0.7456 |
| 10.0 | 0.9025 | 0.0022 | 0.4289 | 0.8626 |
| 15.0 | 0.6867 | 0.0861 | 0.6602 | 0.9380 |
