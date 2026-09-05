# TARGET Domain Multi-Model Performance Synthesis Report

## 1. Bit Error Rate (BER) Comparison

| SNR (dB) | LI+DnCNN CORAL block 3 | LI+DnCNN CORAL block 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.482506 | 0.484857 | 0.479818 | 0.449321 |
| -5.0 | 0.470108 | 0.467242 | 0.441466 | 0.403807 |
| 0.0 | 0.432377 | 0.432998 | 0.357242 | 0.306450 |
| 5.0 | 0.379717 | 0.376938 | 0.233443 | 0.188221 |
| 10.0 | 0.301362 | 0.302701 | 0.113611 | 0.078185 |
| 15.0 | 0.234533 | 0.231030 | 0.030859 | 0.013600 |

## 2. Normalized Mean Squared Error (NMSE) [dB] Comparison

| SNR (dB) | LI+DnCNN CORAL block 3 | LI+DnCNN CORAL block 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 13.51 dB | 13.51 dB | 10.73 dB | -3.78 dB |
| -5.0 | 10.97 dB | 10.72 dB | 5.73 dB | -9.27 dB |
| 0.0 | 7.03 dB | 7.06 dB | 0.74 dB | -13.42 dB |
| 5.0 | 3.22 dB | 3.15 dB | -4.26 dB | -17.23 dB |
| 10.0 | -0.14 dB | -0.10 dB | -9.26 dB | -20.80 dB |
| 15.0 | -2.56 dB | -2.66 dB | -14.27 dB | -24.92 dB |

## 3. Structural Similarity Index (SSIM) Comparison

| SNR (dB) | LI+DnCNN CORAL block 3 | LI+DnCNN CORAL block 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -0.0036 | -0.0063 | 0.0026 | 0.0897 |
| -5.0 | -0.0050 | -0.0101 | 0.0229 | 0.2984 |
| 0.0 | 0.0015 | 0.0094 | 0.0826 | 0.5559 |
| 5.0 | -0.0383 | -0.0114 | 0.2211 | 0.7456 |
| 10.0 | 0.0222 | 0.0485 | 0.4289 | 0.8626 |
| 15.0 | 0.1215 | 0.0024 | 0.6602 | 0.9380 |

## 4. Mean Squared Error (MSE) Comparison

| SNR (dB) | LI+DnCNN CORAL block 3 | LI+DnCNN CORAL block 2,3 | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 2.611e-18 | 2.612e-18 | 3.008e-18 | 1.021e-19 |
| -5.0 | 1.982e-18 | 1.850e-18 | 7.947e-19 | 2.528e-20 |
| 0.0 | 9.423e-19 | 9.441e-19 | 2.643e-19 | 1.016e-20 |
| 5.0 | 4.201e-19 | 4.134e-19 | 8.998e-20 | 4.377e-21 |
| 10.0 | 1.856e-19 | 1.871e-19 | 2.656e-20 | 1.619e-21 |
| 15.0 | 1.182e-19 | 1.154e-19 | 9.644e-21 | 7.557e-22 |
