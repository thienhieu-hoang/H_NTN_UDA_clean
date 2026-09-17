# TARGET Domain Multi-Model Performance Synthesis Report

## 1. Bit Error Rate (BER) Comparison

| SNR (dB) | LI+DnCNN pHead CORAL (block2_block3) | LI+DnCNN pHead CORAL (block3) | LS+Transformer pHead std CORAL (layer1) | LS+Transformer pHead std CORAL (layer1_layer2) | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 0.483141 | 0.484037 | 0.449852 | 0.449507 | 0.479818 | 0.449321 |
| -5.0 | 0.468276 | 0.467753 | 0.401364 | 0.402474 | 0.441466 | 0.403807 |
| 0.0 | 0.431681 | 0.431461 | 0.309030 | 0.309950 | 0.357242 | 0.306450 |
| 5.0 | 0.377058 | 0.372638 | 0.195151 | 0.195872 | 0.233443 | 0.188221 |
| 10.0 | 0.303953 | 0.304169 | 0.090045 | 0.089879 | 0.113611 | 0.078185 |
| 15.0 | 0.228947 | 0.233798 | 0.025521 | 0.025605 | 0.030859 | 0.013600 |

## 2. Normalized Mean Squared Error (NMSE) [dB] Comparison

| SNR (dB) | LI+DnCNN pHead CORAL (block2_block3) | LI+DnCNN pHead CORAL (block3) | LS+Transformer pHead std CORAL (layer1) | LS+Transformer pHead std CORAL (layer1_layer2) | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 13.45 dB | 13.40 dB | -5.94 dB | -6.12 dB | 10.73 dB | -3.78 dB |
| -5.0 | 10.77 dB | 10.81 dB | -9.34 dB | -8.13 dB | 5.73 dB | -9.27 dB |
| 0.0 | 7.22 dB | 7.02 dB | -11.78 dB | -11.32 dB | 0.74 dB | -13.42 dB |
| 5.0 | 3.27 dB | 3.16 dB | -13.40 dB | -13.32 dB | -4.26 dB | -17.23 dB |
| 10.0 | -0.12 dB | -0.02 dB | -15.16 dB | -15.28 dB | -9.26 dB | -20.80 dB |
| 15.0 | -2.73 dB | -2.43 dB | -16.95 dB | -16.89 dB | -14.27 dB | -24.92 dB |

## 3. Structural Similarity Index (SSIM) Comparison

| SNR (dB) | LI+DnCNN pHead CORAL (block2_block3) | LI+DnCNN pHead CORAL (block3) | LS+Transformer pHead std CORAL (layer1) | LS+Transformer pHead std CORAL (layer1_layer2) | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | -0.0014 | -0.0043 | 0.1696 | 0.1496 | 0.0026 | 0.0897 |
| -5.0 | -0.0075 | -0.0042 | 0.2652 | 0.3424 | 0.0229 | 0.2984 |
| 0.0 | 0.0036 | 0.0098 | 0.4604 | 0.4462 | 0.0826 | 0.5559 |
| 5.0 | 0.0058 | -0.0610 | 0.5479 | 0.5629 | 0.2211 | 0.7456 |
| 10.0 | 0.0022 | -0.0553 | 0.6678 | 0.6689 | 0.4289 | 0.8626 |
| 15.0 | 0.0861 | -0.0135 | 0.7494 | 0.7530 | 0.6602 | 0.9380 |

## 4. Mean Squared Error (MSE) Comparison

| SNR (dB) | LI+DnCNN pHead CORAL (block2_block3) | LI+DnCNN pHead CORAL (block3) | LS+Transformer pHead std CORAL (layer1) | LS+Transformer pHead std CORAL (layer1_layer2) | LI Benchmark | MMSE Benchmark |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -10.0 | 2.563e-18 | 2.543e-18 | 7.158e-20 | 6.796e-20 | 3.008e-18 | 1.021e-19 |
| -5.0 | 1.879e-18 | 1.904e-18 | 2.497e-20 | 3.559e-20 | 7.947e-19 | 2.528e-20 |
| 0.0 | 9.874e-19 | 9.365e-19 | 1.528e-20 | 1.680e-20 | 2.643e-19 | 1.016e-20 |
| 5.0 | 4.250e-19 | 4.138e-19 | 1.145e-20 | 1.172e-20 | 8.998e-20 | 4.377e-21 |
| 10.0 | 1.865e-19 | 1.908e-19 | 6.931e-21 | 6.733e-21 | 2.656e-20 | 1.619e-21 |
| 15.0 | 1.147e-19 | 1.211e-19 | 5.842e-21 | 5.907e-21 | 9.644e-21 | 7.557e-22 |
