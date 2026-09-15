# NTN Channel Estimation Synthesis Summary (LS)

## BER Comparison Table
| SNR (dB) | LS+Attention+cGAN std | LMMSE Benchmark | LI Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.455298 | 0.453490 | 0.482474 |
| -5.0 | 0.406738 | 0.407575 | 0.449624 |
| 0.0 | 0.321898 | 0.318218 | 0.380368 |
| 5.0 | 0.206136 | 0.201977 | 0.269748 |
| 10.0 | 0.104120 | 0.099420 | 0.159096 |
| 15.0 | 0.033005 | 0.030274 | 0.067497 |

## NMSE (dB) Comparison Table
| SNR (dB) | LS+Attention+cGAN std | LMMSE Benchmark | LI Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | -3.92 dB | -2.72 dB | 13.05 dB |
| -5.0 | -6.58 dB | -7.32 dB | 8.21 dB |
| 0.0 | -9.97 dB | -11.82 dB | 3.29 dB |
| 5.0 | -14.29 dB | -16.34 dB | -1.93 dB |
| 10.0 | -17.62 dB | -20.59 dB | -6.81 dB |
| 15.0 | -21.81 dB | -25.08 dB | -11.89 dB |

## SSIM Comparison Table
| SNR (dB) | LS+Attention+cGAN std | LMMSE Benchmark | LI Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.4322 | 0.1047 | 0.0065 |
| -5.0 | 0.5273 | 0.4846 | 0.0222 |
| 0.0 | 0.7676 | 0.8190 | 0.1238 |
| 5.0 | 0.8861 | 0.9317 | 0.3249 |
| 10.0 | 0.9431 | 0.9690 | 0.6076 |
| 15.0 | 0.9739 | 0.9870 | 0.7996 |
