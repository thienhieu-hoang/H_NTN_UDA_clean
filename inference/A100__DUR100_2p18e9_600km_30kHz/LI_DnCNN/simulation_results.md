# NTN Inferred Channel Equalization & Performance Summary

## Target Batch Directory
`C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/inference/A100__DUR100_2p18e9_600km_30kHz/LI_DnCNN`

## BER Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.478966 | 0.486779 | 0.449289 |
| -5.0 | 0.440834 | 0.450996 | 0.403469 |
| 0.0 | 0.357410 | 0.367795 | 0.307729 |
| 5.0 | 0.234593 | 0.248976 | 0.189900 |
| 10.0 | 0.115154 | 0.121469 | 0.079528 |
| 15.0 | 0.032287 | 0.031417 | 0.014859 |

## NMSE (dB) Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 10.90 dB | 13.23 dB | -3.90 dB |
| -5.0 | 5.90 dB | 7.09 dB | -9.39 dB |
| 0.0 | 0.90 dB | 1.96 dB | -13.25 dB |
| 5.0 | -4.10 dB | -3.10 dB | -17.00 dB |
| 10.0 | -9.10 dB | -7.94 dB | -20.76 dB |
| 15.0 | -14.09 dB | -13.69 dB | -24.78 dB |

## SSIM Performance Table
| SNR (dB) | LS + Linear Interpolation | LI+DnCNN inferred | MMSE Benchmark |
|:---:|:---:|:---:|:---:|
| -10.0 | 0.0028 | -0.0016 | 0.1001 |
| -5.0 | 0.0208 | 0.0071 | 0.2817 |
| 0.0 | 0.0818 | 0.0640 | 0.5681 |
| 5.0 | 0.2261 | 0.2253 | 0.7547 |
| 10.0 | 0.4422 | 0.4286 | 0.8665 |
| 15.0 | 0.6533 | 0.6414 | 0.9350 |
