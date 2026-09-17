# CORAL UDA Multi-Model Performance Synthesis Report

Consolidated performance comparison across evaluated models/layers for both Target and Source domains.

## TARGET Domain Evaluation

### BER Performance (TARGET)
| SNR (dB) | LI+DnCNN pHead CORAL (block2_block3) | LI+DnCNN pHead CORAL (block3) | LS+Transformer pHead std CORAL (layer1) | LS+Transformer pHead std CORAL (layer1_layer2) | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.483141 | 0.484037 | 0.449852 | 0.449507 | 0.479818 | 0.449321 |
| -5.0 | 0.468276 | 0.467753 | 0.401364 | 0.402474 | 0.441466 | 0.403807 |
| 0.0 | 0.431681 | 0.431461 | 0.309030 | 0.309950 | 0.357242 | 0.306450 |
| 5.0 | 0.377058 | 0.372638 | 0.195151 | 0.195872 | 0.233443 | 0.188221 |
| 10.0 | 0.303953 | 0.304169 | 0.090045 | 0.089879 | 0.113611 | 0.078185 |
| 15.0 | 0.228947 | 0.233798 | 0.025521 | 0.025605 | 0.030859 | 0.013600 |

### NMSE [dB] Performance (TARGET)
| SNR (dB) | LI+DnCNN pHead CORAL (block2_block3) | LI+DnCNN pHead CORAL (block3) | LS+Transformer pHead std CORAL (layer1) | LS+Transformer pHead std CORAL (layer1_layer2) | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 13.45 dB | 13.40 dB | -5.94 dB | -6.12 dB | 10.73 dB | -3.78 dB |
| -5.0 | 10.77 dB | 10.81 dB | -9.34 dB | -8.13 dB | 5.73 dB | -9.27 dB |
| 0.0 | 7.22 dB | 7.02 dB | -11.78 dB | -11.32 dB | 0.74 dB | -13.42 dB |
| 5.0 | 3.27 dB | 3.16 dB | -13.40 dB | -13.32 dB | -4.26 dB | -17.23 dB |
| 10.0 | -0.12 dB | -0.02 dB | -15.16 dB | -15.28 dB | -9.26 dB | -20.80 dB |
| 15.0 | -2.73 dB | -2.43 dB | -16.95 dB | -16.89 dB | -14.27 dB | -24.92 dB |

## SOURCE Domain Evaluation

### BER Performance (SOURCE)
| SNR (dB) | LI+DnCNN pHead CORAL (block2_block3) | LI+DnCNN pHead CORAL (block3) | LS+Transformer pHead std CORAL (layer1) | LS+Transformer pHead std CORAL (layer1_layer2) | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.459511 | 0.459552 | 0.455740 | 0.455745 | 0.482014 | 0.454994 |
| -5.0 | 0.412963 | 0.413578 | 0.412396 | 0.407842 | 0.450269 | 0.411016 |
| 0.0 | 0.327505 | 0.327744 | 0.323945 | 0.324605 | 0.380874 | 0.321271 |
| 5.0 | 0.212092 | 0.228450 | 0.209895 | 0.209459 | 0.271373 | 0.206588 |
| 10.0 | 0.113081 | 0.121582 | 0.106105 | 0.105722 | 0.160584 | 0.103031 |
| 15.0 | 0.084738 | 0.042231 | 0.034247 | 0.034212 | 0.067918 | 0.031753 |

### NMSE [dB] Performance (SOURCE)
| SNR (dB) | LI+DnCNN pHead CORAL (block2_block3) | LI+DnCNN pHead CORAL (block3) | LS+Transformer pHead std CORAL (layer1) | LS+Transformer pHead std CORAL (layer1_layer2) | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -1.84 dB | -2.13 dB | -4.60 dB | -4.58 dB | 13.05 dB | -2.72 dB |
| -5.0 | -5.44 dB | -5.36 dB | -6.46 dB | -7.82 dB | 8.21 dB | -7.32 dB |
| 0.0 | -8.84 dB | -8.87 dB | -10.48 dB | -10.19 dB | 3.29 dB | -11.82 dB |
| 5.0 | -13.43 dB | -8.39 dB | -14.80 dB | -14.74 dB | -1.93 dB | -16.34 dB |
| 10.0 | -15.16 dB | -12.44 dB | -18.18 dB | -18.45 dB | -6.81 dB | -20.59 dB |
| 15.0 | -7.79 dB | -17.00 dB | -21.98 dB | -21.85 dB | -11.89 dB | -25.08 dB |

