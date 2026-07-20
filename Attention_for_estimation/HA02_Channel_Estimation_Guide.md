# HA02 Architecture & Implementation Guide

This guide details the implementation of the **HA02** hybrid attention-convolutional neural network for wireless channel estimation, as presented in the paper *Attention Based Neural Networks for Wireless Channel Estimation* (Dianxin Luan & John Thompson, IEEE VTC2022-Spring).

The corresponding PyTorch implementation is provided in [ha02_channel_estimation.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/Attention_for_estimation/ha02_channel_estimation.py).

---

## 1. System Parameters & Dimension Specifications

| Parameter | Symbol | Value | Notes |
| :--- | :--- | :--- | :--- |
| **OFDM Symbols per Slot** | $N_s$ | `14` | 5G NR slot structure |
| **Subcarriers** | $N_f$ | `72` | 6 Resource Blocks (RBs) |
| **Pilot Symbols** | $N_{pilot}$ | `2` | Reserved symbols (indices 1 & 13) |
| **Pilot Subcarrier Spacing** | - | `2` | $\frac{N_f}{2} = 36$ subcarriers per pilot symbol |
| **Total Pilot Elements** | $N_{pilot} \times \frac{N_f}{2}$ | `72` | Complex-valued LS pilot estimates |
| **Model Input Dimensions** | $\mathbb{R}^{72 \times 2}$ | `(Batch, 72, 2)` | Real and Imaginary components split |
| **Model Output Dimensions**| $\mathbb{R}^{1008 \times 2}$ | `(Batch, 14, 72, 2)` | Reconstructed full 2D channel grid ($14 \times 72 \times 2$) |

---

## 2. Key Architectural Components

```
                    Raw LS Pilot Estimates (72, 2)
                                 │
                                 ▼
           ┌───────────────────────────────────────────┐
           │      Transformer Encoder Stack            │
           │  • FC Projection & Q, K, V Generation     │
           │  • Multi-Head Self-Attention (N_heads=2)  │
           │  • Add & Layer Normalization              │
           │  • Feed-Forward Network (FC-GeLU-FC)      │
           │  • Add & Layer Normalization              │
           └───────────────────────────────────────────┘
                                 │
                        Feature Vector (72, 2)
                                 │
                                 ▼
           ┌───────────────────────────────────────────┐
           │   Residual Convolutional Decoder          │
           │  • Conv1 (2x2 kernel, N_filter=2)         │
           │  • Residual Conv Block                    │
           │  • FC + Conv 1D Upsampling Module         │
           │  • Conv Output Layer                      │
           └───────────────────────────────────────────┘
                                 │
                                 ▼
               Full Grid Channel Estimate (14, 72, 2)
```

### A. Inputs (No Pre-Interpolation)
* Conventional channel estimation networks (like **ChannelNet**) take an interpolated, full-grid 2D noisy matrix as input.
* **HA02** takes **only raw LS pilot estimates** ($\hat{H}_{LS} \in \mathbb{C}^{72}$) without pre-interpolating or zero-padding the non-pilot resource elements.

### B. Encoder: Multi-Head Self-Attention
* **Role:** Pre-processes sparse LS pilot estimates to identify correlations across pilot subcarriers.
* **Mechanism:**
  * Uses $N_{heads} = N_{pilot} = 2$.
  * Projects input to Key ($K$), Query ($Q$), and Value ($V$).
  * Scaled Dot-Product Attention:
    $$\text{Attention} = \text{softmax}\left(\frac{Q K^T}{\sqrt{N_f / 2}}\right) V = \text{softmax}\left(\frac{Q K^T}{6}\right) V$$
  * Passes through Feed-Forward Network with **GeLU** activation and Layer Normalization.

### C. Decoder: ResNet + Upsampling
* **Role:** Simultaneously performs denoising and 2D upsampling/interpolation.
* **Mechanism:**
  * Processed by 2D Convolutional layers and a 1-block Residual module ($N_{filter} = 2$).
  * A Linear/FC layer upsamples the pilot dimension (72) to the full slot dimension ($1008 = 14 \times 72$).
  * Final 2D Conv layer outputs the reconstructed channel grid of shape `(14, 72, 2)`.

---

## 3. Loss Function & Training Parameters

* **Loss Function:** Huber Loss ($\delta = 1.0$)
  $$L_\delta(a) = \begin{cases} \frac{1}{2} a^2 & \text{if } |a| \le \delta \\ \delta(|a| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$
* **Optimizer:** Adam (Initial learning rate = 0.002, dropped by 0.5 every 20 epochs).
* **Batch Size:** 128
* **Epochs:** 100
* **Total Parameters:** ~105,607 parameters (Encoder: ~31.8k, Decoder: ~73.8k).

---

## 4. How to Use the Python Code

The script [ha02_channel_estimation.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/Attention_for_estimation/ha02_channel_estimation.py) includes a dataset loader (`MatChannelDataset`) designed to load `.mat` datasets containing channel realizations.

### Steps to Load Your `.mat` Files:
1. Open [ha02_channel_estimation.py](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/Attention_for_estimation/ha02_channel_estimation.py).
2. Locate the line:
   ```python
   MAT_FILE_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\Attention_for_estimation\mat_data"
   ```
3. Set `MAT_FILE_DIR` to the directory path where your `.mat` files are saved.
4. Ensure your `.mat` files contain the keys:
   * `'H_LS'`: LS channel estimates at pilot positions (shape: `72` complex or `36 x 2`).
   * `'H_true'`: Ground truth full channel matrix (shape: `14 x 72` complex).
5. Run the code to verify model initialization and data loading:
   ```bash
   python ha02_channel_estimation.py
   ```
