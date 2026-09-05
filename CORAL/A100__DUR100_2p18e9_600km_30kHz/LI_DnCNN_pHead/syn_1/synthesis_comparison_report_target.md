# TARGET Domain Multi-Model Performance Synthesis Report

## 1. Bit Error Rate (BER) Comparison

| SNR (dB) | LI+DnCNN CORAL pHead layer 3 | LI+DnCNN CORAL pHead layer 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.484037 | 0.483141 | 0.479818 | 0.449321 |
| -5.0 | 0.467753 | 0.468276 | 0.441466 | 0.403807 |
| 0.0 | 0.431461 | 0.431681 | 0.357242 | 0.306450 |
| 5.0 | 0.372638 | 0.377058 | 0.233443 | 0.188221 |
| 10.0 | 0.304169 | 0.303953 | 0.113611 | 0.078185 |
| 15.0 | 0.233798 | 0.228947 | 0.030859 | 0.013600 |

## 2. Normalized Mean Squared Error (NMSE) [dB] Comparison

| SNR (dB) | LI+DnCNN CORAL pHead layer 3 | LI+DnCNN CORAL pHead layer 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 13.40 dB | 13.45 dB | 10.73 dB | -3.78 dB |
| -5.0 | 10.81 dB | 10.77 dB | 5.73 dB | -9.27 dB |
| 0.0 | 7.02 dB | 7.22 dB | 0.74 dB | -13.42 dB |
| 5.0 | 3.16 dB | 3.27 dB | -4.26 dB | -17.23 dB |
| 10.0 | -0.02 dB | -0.12 dB | -9.26 dB | -20.80 dB |
| 15.0 | -2.43 dB | -2.73 dB | -14.27 dB | -24.92 dB |

## 3. Structural Similarity Index (SSIM) Comparison

| SNR (dB) | LI+DnCNN CORAL pHead layer 3 | LI+DnCNN CORAL pHead layer 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -0.0043 | -0.0014 | 0.0026 | 0.0897 |
| -5.0 | -0.0042 | -0.0075 | 0.0229 | 0.2984 |
| 0.0 | 0.0098 | 0.0036 | 0.0826 | 0.5559 |
| 5.0 | -0.0610 | 0.0058 | 0.2211 | 0.7456 |
| 10.0 | -0.0553 | 0.0022 | 0.4289 | 0.8626 |
| 15.0 | -0.0135 | 0.0861 | 0.6602 | 0.9380 |

## 4. Mean Squared Error (MSE) Comparison

| SNR (dB) | LI+DnCNN CORAL pHead layer 3 | LI+DnCNN CORAL pHead layer 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 2.543e-18 | 2.563e-18 | 3.008e-18 | 1.021e-19 |
| -5.0 | 1.904e-18 | 1.879e-18 | 7.947e-19 | 2.528e-20 |
| 0.0 | 9.365e-19 | 9.874e-19 | 2.643e-19 | 1.016e-20 |
| 5.0 | 4.138e-19 | 4.250e-19 | 8.998e-20 | 4.377e-21 |
| 10.0 | 1.908e-19 | 1.865e-19 | 2.656e-20 | 1.619e-21 |
| 15.0 | 1.211e-19 | 1.147e-19 | 9.644e-21 | 7.557e-22 |
