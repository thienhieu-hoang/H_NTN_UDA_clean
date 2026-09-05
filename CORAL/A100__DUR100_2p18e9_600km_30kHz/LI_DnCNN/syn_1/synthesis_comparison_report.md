# CORAL UDA Multi-Model Performance Synthesis Report

Consolidated performance comparison across evaluated models/layers for both Target and Source domains.

## TARGET Domain Evaluation

### BER Performance (TARGET)
| SNR (dB) | LI+DnCNN CORAL layer 3 | LI+DnCNN CORAL layer 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.482506 | 0.484857 | 0.479818 | 0.449321 |
| -5.0 | 0.470108 | 0.467242 | 0.441466 | 0.403807 |
| 0.0 | 0.432377 | 0.432998 | 0.357242 | 0.306450 |
| 5.0 | 0.379717 | 0.376938 | 0.233443 | 0.188221 |
| 10.0 | 0.301362 | 0.302701 | 0.113611 | 0.078185 |
| 15.0 | 0.234533 | 0.231030 | 0.030859 | 0.013600 |

### NMSE [dB] Performance (TARGET)
| SNR (dB) | LI+DnCNN CORAL layer 3 | LI+DnCNN CORAL layer 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 13.51 dB | 13.51 dB | 10.73 dB | -3.78 dB |
| -5.0 | 10.97 dB | 10.72 dB | 5.73 dB | -9.27 dB |
| 0.0 | 7.03 dB | 7.06 dB | 0.74 dB | -13.42 dB |
| 5.0 | 3.22 dB | 3.15 dB | -4.26 dB | -17.23 dB |
| 10.0 | -0.14 dB | -0.10 dB | -9.26 dB | -20.80 dB |
| 15.0 | -2.56 dB | -2.66 dB | -14.27 dB | -24.92 dB |

## SOURCE Domain Evaluation

### BER Performance (SOURCE)
| SNR (dB) | LI+DnCNN CORAL layer 3 | LI+DnCNN CORAL layer 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.459441 | 0.459796 | 0.482014 | 0.454994 |
| -5.0 | 0.412855 | 0.413127 | 0.450269 | 0.411016 |
| 0.0 | 0.326954 | 0.328442 | 0.380874 | 0.321271 |
| 5.0 | 0.213785 | 0.212888 | 0.271373 | 0.206588 |
| 10.0 | 0.112244 | 0.116580 | 0.160584 | 0.103031 |
| 15.0 | 0.043575 | 0.039352 | 0.067918 | 0.031753 |

### NMSE [dB] Performance (SOURCE)
| SNR (dB) | LI+DnCNN CORAL layer 3 | LI+DnCNN CORAL layer 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -2.35 dB | -1.98 dB | 13.05 dB | -2.72 dB |
| -5.0 | -5.48 dB | -5.55 dB | 8.21 dB | -7.32 dB |
| 0.0 | -9.06 dB | -8.67 dB | 3.29 dB | -11.82 dB |
| 5.0 | -12.61 dB | -13.04 dB | -1.93 dB | -16.34 dB |
| 10.0 | -15.32 dB | -14.00 dB | -6.81 dB | -20.59 dB |
| 15.0 | -16.58 dB | -18.91 dB | -11.89 dB | -25.08 dB |

