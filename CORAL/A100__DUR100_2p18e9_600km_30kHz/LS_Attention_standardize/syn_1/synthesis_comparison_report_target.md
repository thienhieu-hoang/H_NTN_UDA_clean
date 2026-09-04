# TARGET Domain Multi-Model Performance Synthesis Report

## 1. Bit Error Rate (BER) Comparison

| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.451271 | 0.451259 | 0.479818 | 0.449321 |
| -5.0 | 0.403240 | 0.403124 | 0.441466 | 0.403807 |
| 0.0 | 0.311546 | 0.311581 | 0.357242 | 0.306450 |
| 5.0 | 0.195093 | 0.196897 | 0.233443 | 0.188221 |
| 10.0 | 0.090398 | 0.091143 | 0.113611 | 0.078185 |
| 15.0 | 0.027974 | 0.030800 | 0.030859 | 0.013600 |

## 2. Normalized Mean Squared Error (NMSE) [dB] Comparison

| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -4.83 dB | -4.68 dB | 10.73 dB | -3.78 dB |
| -5.0 | -7.56 dB | -7.50 dB | 5.73 dB | -9.27 dB |
| 0.0 | -10.62 dB | -10.39 dB | 0.74 dB | -13.42 dB |
| 5.0 | -13.55 dB | -12.79 dB | -4.26 dB | -17.23 dB |
| 10.0 | -15.05 dB | -14.87 dB | -9.26 dB | -20.80 dB |
| 15.0 | -16.06 dB | -15.16 dB | -14.27 dB | -24.92 dB |

## 3. Structural Similarity Index (SSIM) Comparison

| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.1552 | 0.1784 | 0.0026 | 0.0897 |
| -5.0 | 0.3140 | 0.3191 | 0.0229 | 0.2984 |
| 0.0 | 0.4342 | 0.4261 | 0.0826 | 0.5559 |
| 5.0 | 0.5629 | 0.5440 | 0.2211 | 0.7456 |
| 10.0 | 0.6650 | 0.6597 | 0.4289 | 0.8626 |
| 15.0 | 0.7295 | 0.7032 | 0.6602 | 0.9380 |

## 4. Mean Squared Error (MSE) Comparison

| SNR (dB) | LS+Transformer CORAL layer 1 | LS+Transformer CORAL layer 1,2 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 9.539e-20 | 9.641e-20 | 3.008e-18 | 1.021e-19 |
| -5.0 | 3.992e-20 | 4.145e-20 | 7.947e-19 | 2.528e-20 |
| 0.0 | 2.044e-20 | 2.147e-20 | 2.643e-19 | 1.016e-20 |
| 5.0 | 1.133e-20 | 1.341e-20 | 8.998e-20 | 4.377e-21 |
| 10.0 | 7.061e-21 | 7.406e-21 | 2.656e-20 | 1.619e-21 |
| 15.0 | 6.543e-21 | 7.321e-21 | 9.644e-21 | 7.557e-22 |
