# NTN Channel Estimation Synthesis Summary (LS)

## BER Comparison Table
| SNR (dB) | LS+Attention+UNet std | LMMSE Benchmark | LI Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.453520 | 0.453490 | 0.482474 |
| -5.0 | 0.406147 | 0.407575 | 0.449624 |
| 0.0 | 0.320752 | 0.318218 | 0.380368 |
| 5.0 | 0.204983 | 0.201977 | 0.269748 |
| 10.0 | 0.102158 | 0.099420 | 0.159096 |
| 15.0 | 0.032737 | 0.030274 | 0.067497 |

## NMSE (dB) Comparison Table
| SNR (dB) | LS+Attention+UNet std | LMMSE Benchmark | LI Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | -4.62 dB | -2.72 dB | 13.05 dB |
| -5.0 | -7.16 dB | -7.32 dB | 8.21 dB |
| 0.0 | -10.48 dB | -11.82 dB | 3.29 dB |
| 5.0 | -14.94 dB | -16.34 dB | -1.93 dB |
| 10.0 | -18.68 dB | -20.59 dB | -6.81 dB |
| 15.0 | -22.07 dB | -25.08 dB | -11.89 dB |

## SSIM Comparison Table
| SNR (dB) | LS+Attention+UNet std | LMMSE Benchmark | LI Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.3241 | 0.1047 | 0.0065 |
| -5.0 | 0.4950 | 0.4846 | 0.0222 |
| 0.0 | 0.7752 | 0.8190 | 0.1238 |
| 5.0 | 0.9019 | 0.9317 | 0.3249 |
| 10.0 | 0.9511 | 0.9690 | 0.6076 |
| 15.0 | 0.9766 | 0.9870 | 0.7996 |
