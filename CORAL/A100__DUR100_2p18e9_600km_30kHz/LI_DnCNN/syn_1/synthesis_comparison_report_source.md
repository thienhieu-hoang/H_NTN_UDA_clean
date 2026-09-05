# SOURCE Domain Multi-Model Performance Synthesis Report

## 1. Bit Error Rate (BER) Comparison

| SNR (dB) | LI+DnCNN CORAL layer 3 | LI+DnCNN CORAL layer 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.459441 | 0.459796 | 0.482014 | 0.454994 |
| -5.0 | 0.412855 | 0.413127 | 0.450269 | 0.411016 |
| 0.0 | 0.326954 | 0.328442 | 0.380874 | 0.321271 |
| 5.0 | 0.213785 | 0.212888 | 0.271373 | 0.206588 |
| 10.0 | 0.112244 | 0.116580 | 0.160584 | 0.103031 |
| 15.0 | 0.043575 | 0.039352 | 0.067918 | 0.031753 |

## 2. Normalized Mean Squared Error (NMSE) [dB] Comparison

| SNR (dB) | LI+DnCNN CORAL layer 3 | LI+DnCNN CORAL layer 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -2.35 dB | -1.98 dB | 13.05 dB | -2.72 dB |
| -5.0 | -5.48 dB | -5.55 dB | 8.21 dB | -7.32 dB |
| 0.0 | -9.06 dB | -8.67 dB | 3.29 dB | -11.82 dB |
| 5.0 | -12.61 dB | -13.04 dB | -1.93 dB | -16.34 dB |
| 10.0 | -15.32 dB | -14.00 dB | -6.81 dB | -20.59 dB |
| 15.0 | -16.58 dB | -18.91 dB | -11.89 dB | -25.08 dB |

## 3. Structural Similarity Index (SSIM) Comparison

| SNR (dB) | LI+DnCNN CORAL layer 3 | LI+DnCNN CORAL layer 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.2116 | 0.1703 | 0.0065 | 0.1047 |
| -5.0 | 0.3974 | 0.3811 | 0.0222 | 0.4846 |
| 0.0 | 0.6842 | 0.6793 | 0.1238 | 0.8190 |
| 5.0 | 0.8262 | 0.8377 | 0.3249 | 0.9317 |
| 10.0 | 0.9056 | 0.8727 | 0.6076 | 0.9690 |
| 15.0 | 0.9249 | 0.9515 | 0.7996 | 0.9870 |

## 4. Mean Squared Error (MSE) Comparison

| SNR (dB) | LI+DnCNN CORAL layer 3 | LI+DnCNN CORAL layer 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 1.486e-16 | 1.622e-16 | 5.649e-15 | 1.361e-16 |
| -5.0 | 7.748e-17 | 7.505e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 3.295e-17 | 3.654e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 1.508e-17 | 1.366e-17 | 1.997e-16 | 5.395e-18 |
| 10.0 | 7.713e-18 | 1.078e-17 | 6.144e-17 | 2.211e-18 |
| 15.0 | 6.193e-18 | 3.578e-18 | 2.133e-17 | 8.671e-19 |
