# SOURCE Domain Multi-Model Performance Synthesis Report

## 1. Bit Error Rate (BER) Comparison

| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.455740 | 0.455745 | 0.482014 | 0.454994 |
| -5.0 | 0.412396 | 0.407842 | 0.450269 | 0.411016 |
| 0.0 | 0.323945 | 0.324605 | 0.380874 | 0.321271 |
| 5.0 | 0.209895 | 0.209459 | 0.271373 | 0.206588 |
| 10.0 | 0.106105 | 0.105722 | 0.160584 | 0.103031 |
| 15.0 | 0.034247 | 0.034212 | 0.067918 | 0.031753 |

## 2. Normalized Mean Squared Error (NMSE) [dB] Comparison

| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -4.60 dB | -4.58 dB | 13.05 dB | -2.72 dB |
| -5.0 | -6.46 dB | -7.82 dB | 8.21 dB | -7.32 dB |
| 0.0 | -10.48 dB | -10.19 dB | 3.29 dB | -11.82 dB |
| 5.0 | -14.80 dB | -14.74 dB | -1.93 dB | -16.34 dB |
| 10.0 | -18.18 dB | -18.45 dB | -6.81 dB | -20.59 dB |
| 15.0 | -21.98 dB | -21.85 dB | -11.89 dB | -25.08 dB |

## 3. Structural Similarity Index (SSIM) Comparison

| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.3201 | 0.2811 | 0.0065 | 0.1047 |
| -5.0 | 0.3562 | 0.5707 | 0.0222 | 0.4846 |
| 0.0 | 0.7666 | 0.7434 | 0.1238 | 0.8190 |
| 5.0 | 0.8971 | 0.8966 | 0.3249 | 0.9317 |
| 10.0 | 0.9519 | 0.9532 | 0.6076 | 0.9690 |
| 15.0 | 0.9773 | 0.9756 | 0.7996 | 0.9870 |

## 4. Mean Squared Error (MSE) Comparison

| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 8.200e-17 | 8.114e-17 | 5.649e-15 | 1.361e-16 |
| -5.0 | 5.524e-17 | 4.458e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 2.046e-17 | 2.226e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 9.366e-18 | 8.896e-18 | 1.997e-16 | 5.395e-18 |
| 10.0 | 3.645e-18 | 3.666e-18 | 6.144e-17 | 2.211e-18 |
| 15.0 | 1.551e-18 | 1.641e-18 | 2.133e-17 | 8.671e-19 |
