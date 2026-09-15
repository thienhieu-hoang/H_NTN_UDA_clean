# NTN Channel Estimation Synthesis Summary (LS)

## BER Comparison Table
| SNR (dB) | LS+Attention+Residual | LMMSE Benchmark | LI Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.455477 | 0.453490 | 0.482474 |
| -5.0 | 0.407356 | 0.407575 | 0.449624 |
| 0.0 | 0.322863 | 0.318218 | 0.380368 |
| 5.0 | 0.207593 | 0.201977 | 0.269748 |
| 10.0 | 0.104358 | 0.099420 | 0.159096 |
| 15.0 | 0.033551 | 0.030274 | 0.067497 |

## NMSE (dB) Comparison Table
| SNR (dB) | LS+Attention+Residual | LMMSE Benchmark | LI Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | -3.56 dB | -2.72 dB | 13.05 dB |
| -5.0 | -6.37 dB | -7.32 dB | 8.21 dB |
| 0.0 | -9.81 dB | -11.82 dB | 3.29 dB |
| 5.0 | -13.54 dB | -16.34 dB | -1.93 dB |
| 10.0 | -17.67 dB | -20.59 dB | -6.81 dB |
| 15.0 | -20.98 dB | -25.08 dB | -11.89 dB |

## SSIM Comparison Table
| SNR (dB) | LS+Attention+Residual | LMMSE Benchmark | LI Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.3405 | 0.1047 | 0.0065 |
| -5.0 | 0.5252 | 0.4846 | 0.0222 |
| 0.0 | 0.7703 | 0.8190 | 0.1238 |
| 5.0 | 0.8695 | 0.9317 | 0.3249 |
| 10.0 | 0.9468 | 0.9690 | 0.6076 |
| 15.0 | 0.9714 | 0.9870 | 0.7996 |
