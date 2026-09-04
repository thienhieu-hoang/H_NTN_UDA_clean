# SOURCE Domain Multi-Model Performance Synthesis Report

## 1. Bit Error Rate (BER) Comparison

| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.456137 | 0.455658 | 0.482014 | 0.454994 |
| -5.0 | 0.408746 | 0.408812 | 0.450269 | 0.411016 |
| 0.0 | 0.323213 | 0.322950 | 0.380874 | 0.321271 |
| 5.0 | 0.209523 | 0.210115 | 0.271373 | 0.206588 |
| 10.0 | 0.106542 | 0.106723 | 0.160584 | 0.103031 |
| 15.0 | 0.034405 | 0.033779 | 0.067918 | 0.031753 |

## 2. Normalized Mean Squared Error (NMSE) [dB] Comparison

| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -4.30 dB | -4.28 dB | 13.05 dB | -2.72 dB |
| -5.0 | -7.09 dB | -7.19 dB | 8.21 dB | -7.32 dB |
| 0.0 | -10.93 dB | -10.78 dB | 3.29 dB | -11.82 dB |
| 5.0 | -14.90 dB | -14.49 dB | -1.93 dB | -16.34 dB |
| 10.0 | -18.34 dB | -18.20 dB | -6.81 dB | -20.59 dB |
| 15.0 | -22.02 dB | -22.27 dB | -11.89 dB | -25.08 dB |

## 3. Structural Similarity Index (SSIM) Comparison

| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.3812 | 0.3764 | 0.0065 | 0.1047 |
| -5.0 | 0.5408 | 0.5544 | 0.0222 | 0.4846 |
| 0.0 | 0.8012 | 0.7893 | 0.1238 | 0.8190 |
| 5.0 | 0.9011 | 0.8965 | 0.3249 | 0.9317 |
| 10.0 | 0.9552 | 0.9534 | 0.6076 | 0.9690 |
| 15.0 | 0.9758 | 0.9769 | 0.7996 | 0.9870 |

## 4. Mean Squared Error (MSE) Comparison

| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 9.652e-17 | 9.492e-17 | 5.649e-15 | 1.361e-16 |
| -5.0 | 5.461e-17 | 5.328e-17 | 1.879e-15 | 4.558e-17 |
| 0.0 | 2.159e-17 | 2.232e-17 | 6.150e-16 | 1.497e-17 |
| 5.0 | 9.061e-18 | 1.010e-17 | 1.997e-16 | 5.395e-18 |
| 10.0 | 3.887e-18 | 4.134e-18 | 6.144e-17 | 2.211e-18 |
| 15.0 | 1.766e-18 | 1.668e-18 | 2.133e-17 | 8.671e-19 |
