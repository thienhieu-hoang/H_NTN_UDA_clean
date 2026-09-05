# CORAL UDA Multi-Model Performance Synthesis Report

Consolidated performance comparison across evaluated models/layers for both Target and Source domains.

## TARGET Domain Evaluation

### BER Performance (TARGET)
| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.449852 | 0.449507 | 0.479818 | 0.449321 |
| -5.0 | 0.401364 | 0.402474 | 0.441466 | 0.403807 |
| 0.0 | 0.309030 | 0.309950 | 0.357242 | 0.306450 |
| 5.0 | 0.195151 | 0.195872 | 0.233443 | 0.188221 |
| 10.0 | 0.090045 | 0.089879 | 0.113611 | 0.078185 |
| 15.0 | 0.025521 | 0.025605 | 0.030859 | 0.013600 |

### NMSE [dB] Performance (TARGET)
| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -5.94 dB | -6.12 dB | 10.73 dB | -3.78 dB |
| -5.0 | -9.34 dB | -8.13 dB | 5.73 dB | -9.27 dB |
| 0.0 | -11.78 dB | -11.32 dB | 0.74 dB | -13.42 dB |
| 5.0 | -13.40 dB | -13.32 dB | -4.26 dB | -17.23 dB |
| 10.0 | -15.16 dB | -15.28 dB | -9.26 dB | -20.80 dB |
| 15.0 | -16.95 dB | -16.89 dB | -14.27 dB | -24.92 dB |

## SOURCE Domain Evaluation

### BER Performance (SOURCE)
| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.455740 | 0.455745 | 0.482014 | 0.454994 |
| -5.0 | 0.412396 | 0.407842 | 0.450269 | 0.411016 |
| 0.0 | 0.323945 | 0.324605 | 0.380874 | 0.321271 |
| 5.0 | 0.209895 | 0.209459 | 0.271373 | 0.206588 |
| 10.0 | 0.106105 | 0.105722 | 0.160584 | 0.103031 |
| 15.0 | 0.034247 | 0.034212 | 0.067918 | 0.031753 |

### NMSE [dB] Performance (SOURCE)
| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -4.60 dB | -4.58 dB | 13.05 dB | -2.72 dB |
| -5.0 | -6.46 dB | -7.82 dB | 8.21 dB | -7.32 dB |
| 0.0 | -10.48 dB | -10.19 dB | 3.29 dB | -11.82 dB |
| 5.0 | -14.80 dB | -14.74 dB | -1.93 dB | -16.34 dB |
| 10.0 | -18.18 dB | -18.45 dB | -6.81 dB | -20.59 dB |
| 15.0 | -21.98 dB | -21.85 dB | -11.89 dB | -25.08 dB |

