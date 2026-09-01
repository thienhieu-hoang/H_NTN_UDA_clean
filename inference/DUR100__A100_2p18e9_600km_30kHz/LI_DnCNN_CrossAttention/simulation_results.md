# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:\\Users\\AT30890\\Hoctap\\1_Hprediction\\working\\H_predict_NTN\\Hest_NTN_UDA_clean\\inference\\DUR100__A100_2p18e9_600km_30kHz\\LI_DnCNN_CrossAttention`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.482047 | 0.486087 | 0.454276 |
| -5.0 | 0.449682 | 0.457272 | 0.411011 |
| 0.0 | 0.380375 | 0.394616 | 0.320705 |
| 5.0 | 0.273889 | 0.275977 | 0.207732 |
| 10.0 | 0.160101 | 0.149622 | 0.101399 |
| 15.0 | 0.068865 | 0.056614 | 0.032343 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 13.15 dB | 10.74 dB | -2.69 dB |
| -5.0 | 8.21 dB | 5.70 dB | -7.36 dB |
| 0.0 | 3.25 dB | 2.22 dB | -11.91 dB |
| 5.0 | -1.84 dB | -2.63 dB | -16.38 dB |
| 10.0 | -6.79 dB | -8.54 dB | -20.72 dB |
| 15.0 | -11.82 dB | -14.20 dB | -25.13 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0047 | -0.0040 | 0.1009 |
| -5.0 | 0.0248 | -0.0232 | 0.5012 |
| 0.0 | 0.1196 | -0.0230 | 0.8124 |
| 5.0 | 0.3379 | 0.1854 | 0.9283 |
| 10.0 | 0.6121 | 0.6106 | 0.9714 |
| 15.0 | 0.8087 | 0.8501 | 0.9885 |
