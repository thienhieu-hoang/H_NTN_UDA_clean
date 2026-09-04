# CORAL UDA Channel Equalization & Domain Performance Summary

**Batch Directory:** `c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/CORAL/A100__DUR100_2p18e9_600km_30kHz/LS_Attention_standardize/layer1_layer2`  
**Model Evaluation:** `LS+Transformer CORAL layer 1,2`  

## Multi-Domain BER Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.455658 | 0.451259 | 0.479818 | 0.449321 |
| -5.0 | 0.408812 | 0.403124 | 0.441466 | 0.403807 |
| 0.0 | 0.322950 | 0.311581 | 0.357242 | 0.306450 |
| 5.0 | 0.210115 | 0.196897 | 0.233443 | 0.188221 |
| 10.0 | 0.106723 | 0.091143 | 0.113611 | 0.078185 |
| 15.0 | 0.033779 | 0.030800 | 0.030859 | 0.013600 |

## Multi-Domain NMSE (dB) Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -4.28 dB | -4.68 dB | 10.73 dB | -3.78 dB |
| -5.0 | -7.19 dB | -7.50 dB | 5.73 dB | -9.27 dB |
| 0.0 | -10.78 dB | -10.39 dB | 0.74 dB | -13.42 dB |
| 5.0 | -14.49 dB | -12.79 dB | -4.26 dB | -17.23 dB |
| 10.0 | -18.20 dB | -14.87 dB | -9.26 dB | -20.80 dB |
| 15.0 | -22.27 dB | -15.16 dB | -14.27 dB | -24.92 dB |

## Multi-Domain SSIM Comparison Table
| SNR (dB) | Source Inferred | Target Inferred | Target LI Benchmark | Target MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.3764 | 0.1784 | 0.0026 | 0.0897 |
| -5.0 | 0.5544 | 0.3191 | 0.0229 | 0.2984 |
| 0.0 | 0.7893 | 0.4261 | 0.0826 | 0.5559 |
| 5.0 | 0.8965 | 0.5440 | 0.2211 | 0.7456 |
| 10.0 | 0.9534 | 0.6597 | 0.4289 | 0.8626 |
| 15.0 | 0.9769 | 0.7032 | 0.6602 | 0.9380 |
