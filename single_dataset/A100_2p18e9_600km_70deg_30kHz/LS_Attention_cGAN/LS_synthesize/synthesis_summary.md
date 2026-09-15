# NTN Channel Estimation Synthesis Summary (LS)

## BER Comparison Table
| SNR (dB) | LS+Attention+cGAN | LMMSE Benchmark | LI Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.456315 | 0.453490 | 0.482474 |
| -5.0 | 0.408020 | 0.407575 | 0.449624 |
| 0.0 | 0.323354 | 0.318218 | 0.380368 |
| 5.0 | 0.208529 | 0.201977 | 0.269748 |
| 10.0 | 0.106363 | 0.099420 | 0.159096 |
| 15.0 | 0.034493 | 0.030274 | 0.067497 |

## NMSE (dB) Comparison Table
| SNR (dB) | LS+Attention+cGAN | LMMSE Benchmark | LI Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | -3.34 dB | -2.72 dB | 13.05 dB |
| -5.0 | -6.08 dB | -7.32 dB | 8.21 dB |
| 0.0 | -9.57 dB | -11.82 dB | 3.29 dB |
| 5.0 | -13.49 dB | -16.34 dB | -1.93 dB |
| 10.0 | -16.68 dB | -20.59 dB | -6.81 dB |
| 15.0 | -20.56 dB | -25.08 dB | -11.89 dB |

## SSIM Comparison Table
| SNR (dB) | LS+Attention+cGAN | LMMSE Benchmark | LI Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.2979 | 0.1047 | 0.0065 |
| -5.0 | 0.4970 | 0.4846 | 0.0222 |
| 0.0 | 0.7592 | 0.8190 | 0.1238 |
| 5.0 | 0.8670 | 0.9317 | 0.3249 |
| 10.0 | 0.9400 | 0.9690 | 0.6076 |
| 15.0 | 0.9680 | 0.9870 | 0.7996 |
