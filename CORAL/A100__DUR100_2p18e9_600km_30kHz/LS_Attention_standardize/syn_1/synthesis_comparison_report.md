# CORAL UDA Multi-Model Performance Synthesis Report

Consolidated performance comparison across evaluated models/layers for both Target and Source domains.

## TARGET Domain Evaluation

### BER Performance (TARGET)
| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.451271 | 0.451259 | 0.479818 | 0.449321 |
| -5.0 | 0.403240 | 0.403124 | 0.441466 | 0.403807 |
| 0.0 | 0.311546 | 0.311581 | 0.357242 | 0.306450 |
| 5.0 | 0.195093 | 0.196897 | 0.233443 | 0.188221 |
| 10.0 | 0.090398 | 0.091143 | 0.113611 | 0.078185 |
| 15.0 | 0.027974 | 0.030800 | 0.030859 | 0.013600 |

### NMSE [dB] Performance (TARGET)
| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -4.83 dB | -4.68 dB | 10.73 dB | -3.78 dB |
| -5.0 | -7.56 dB | -7.50 dB | 5.73 dB | -9.27 dB |
| 0.0 | -10.62 dB | -10.39 dB | 0.74 dB | -13.42 dB |
| 5.0 | -13.55 dB | -12.79 dB | -4.26 dB | -17.23 dB |
| 10.0 | -15.05 dB | -14.87 dB | -9.26 dB | -20.80 dB |
| 15.0 | -16.06 dB | -15.16 dB | -14.27 dB | -24.92 dB |

## SOURCE Domain Evaluation

### BER Performance (SOURCE)
| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.456137 | 0.455658 | 0.482014 | 0.454994 |
| -5.0 | 0.408746 | 0.408812 | 0.450269 | 0.411016 |
| 0.0 | 0.323213 | 0.322950 | 0.380874 | 0.321271 |
| 5.0 | 0.209523 | 0.210115 | 0.271373 | 0.206588 |
| 10.0 | 0.106542 | 0.106723 | 0.160584 | 0.103031 |
| 15.0 | 0.034405 | 0.033779 | 0.067918 | 0.031753 |

### NMSE [dB] Performance (SOURCE)
| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -4.30 dB | -4.28 dB | 13.05 dB | -2.72 dB |
| -5.0 | -7.09 dB | -7.19 dB | 8.21 dB | -7.32 dB |
| 0.0 | -10.93 dB | -10.78 dB | 3.29 dB | -11.82 dB |
| 5.0 | -14.90 dB | -14.49 dB | -1.93 dB | -16.34 dB |
| 10.0 | -18.34 dB | -18.20 dB | -6.81 dB | -20.59 dB |
| 15.0 | -22.02 dB | -22.27 dB | -11.89 dB | -25.08 dB |

