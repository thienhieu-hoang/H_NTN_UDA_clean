# SOURCE Domain Multi-Model Performance Synthesis Report

## 1. Bit Error Rate (BER) Comparison

| SNR (dB) | LI+DnCNN pHead CORAL (block2_block3) | LI+DnCNN pHead CORAL (block3) | LS+Transformer pHead std CORAL (layer1) | LS+Transformer pHead std CORAL (layer1_layer2) | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.459511 | 0.459552 | 0.455740 | 0.455745 | 0.482014 | 0.454994 |
| -5.0 | 0.412963 | 0.413578 | 0.412396 | 0.407842 | 0.450269 | 0.411016 |
| 0.0 | 0.327505 | 0.327744 | 0.323945 | 0.324605 | 0.380874 | 0.321271 |
| 5.0 | 0.212092 | 0.228450 | 0.209895 | 0.209459 | 0.271373 | 0.206588 |
| 10.0 | 0.113081 | 0.121582 | 0.106105 | 0.105722 | 0.160584 | 0.103031 |
| 15.0 | 0.084738 | 0.042231 | 0.034247 | 0.034212 | 0.067918 | 0.031753 |

## 2. Normalized Mean Squared Error (NMSE) [dB] Comparison

| SNR (dB) | LI+DnCNN pHead CORAL (block2_block3) | LI+DnCNN pHead CORAL (block3) | LS+Transformer pHead std CORAL (layer1) | LS+Transformer pHead std CORAL (layer1_layer2) | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -1.84 dB | -2.13 dB | -4.60 dB | -4.58 dB | 13.05 dB | -2.72 dB |
| -5.0 | -5.44 dB | -5.36 dB | -6.46 dB | -7.82 dB | 8.21 dB | -7.32 dB |
| 0.0 | -8.84 dB | -8.87 dB | -10.48 dB | -10.19 dB | 3.29 dB | -11.82 dB |
| 5.0 | -13.43 dB | -8.39 dB | -14.80 dB | -14.74 dB | -1.93 dB | -16.34 dB |
| 10.0 | -15.16 dB | -12.44 dB | -18.18 dB | -18.45 dB | -6.81 dB | -20.59 dB |
| 15.0 | -7.79 dB | -17.00 dB | -21.98 dB | -21.85 dB | -11.89 dB | -25.08 dB |

## 3. Structural Similarity Index (SSIM) Comparison

| SNR (dB) | LI+DnCNN pHead CORAL (block2_block3) | LI+DnCNN pHead CORAL (block3) | LS+Transformer pHead std CORAL (layer1) | LS+Transformer pHead std CORAL (layer1_layer2) | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1979 | 0.1831 | 0.3201 | 0.2811 | 0.0065 | 0.1047 |
| -5.0 | 0.3840 | 0.3778 | 0.3562 | 0.5707 | 0.0222 | 0.4846 |
| 0.0 | 0.6773 | 0.6833 | 0.7666 | 0.7434 | 0.1238 | 0.8190 |
| 5.0 | 0.8479 | 0.6478 | 0.8971 | 0.8966 | 0.3249 | 0.9317 |
| 10.0 | 0.9025 | 0.8389 | 0.9519 | 0.9532 | 0.6076 | 0.9690 |
| 15.0 | 0.6867 | 0.9300 | 0.9773 | 0.9756 | 0.7996 | 0.9870 |

## 4. Mean Squared Error (MSE) Comparison

| SNR (dB) | LI+DnCNN pHead CORAL (block2_block3) | LI+DnCNN pHead CORAL (block3) | LS+Transformer pHead std CORAL (layer1) | LS+Transformer pHead std CORAL (layer1_layer2) | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.679e-16 | 1.551e-16 | 8.200e-17 | 8.114e-17 | 5.649e-15 | 1.361e-16 |
| -5.0 | 7.735e-17 | 7.897e-17 | 5.524e-17 | 4.458e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 3.465e-17 | 3.472e-17 | 2.046e-17 | 2.226e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 1.247e-17 | 4.091e-17 | 9.366e-18 | 8.896e-18 | 1.997e-16 | 5.395e-18 |
| 10.0 | 8.261e-18 | 1.536e-17 | 3.645e-18 | 3.666e-18 | 6.144e-17 | 2.211e-18 |
| 15.0 | 4.283e-17 | 5.450e-18 | 1.551e-18 | 1.641e-18 | 2.133e-17 | 8.671e-19 |
