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


---

Viewed ha02_channel_estimation.py:4-24

Here is a clear, step-by-step breakdown answering all your questions about the workflow, the number 72, the role of attention/transformer, and the decoder:

---

### 1. What does the number 72 represent? Is it 72 subcarriers?

- **Why 72?** In the paper's 5G OFDM slot setup:
  - Total slot size: **14 OFDM symbols** $\times$ **72 subcarriers** = **1008 total positions**.
  - **Pilots:** There are **2 pilot symbols** in the slot. On those 2 pilot symbols, **36 pilot subcarriers** are measured.
  - $2 \text{ pilot symbols} \times 36 \text{ subcarriers} = \mathbf{72 \text{ pilot elements total}}$.
- **Input shape `(72, 2)`:**
  - `72`: The 72 sparse pilot positions in the slot.
  - `2`: Real part & Imaginary part of the complex Least-Squares (LS) estimate $\hat{H}_{\text{LS}} = H_{\text{real}} + j H_{\text{imag}}$.

---

### 2. Complete Step-by-Step Workflow & Dense Layers

Here is the exact data flow through the model:

```
                               INPUT: (Batch, 72, 2)
                                         │
                                   Flatten to (B, 144)
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ 1. TRANSFORMER ENCODER (Attention Pre-processor)                             │
 │                                                                               │
 │  a) Dense Layer (fc1): Maps 144 -> 432 (3 x 144)                              │
 │     --> Output is split into Query (Q), Key (K), Value (V), each of (B, 2, 72)│
 │                                                                               │
 │  b) Self-Attention:                                                           │
 │     --> Computes Pairwise Attention Scores: Softmax(Q @ K^T / sqrt(6)) @ V    │
 │     --> Output shape: (B, 2, 72) -> Flatten back to (B, 144)                 │
 │                                                                               │
 │  c) Dense Layer (fc2): Maps 144 -> 144 (combines multi-head attention)        │
 │  d) Add & LayerNorm 1: Norm(Input + Attention_Output)                         │
 │                                                                               │
 │  e) Feed-Forward Network (FFN):                                               │
 │     --> Dense 1 (144 -> 288) + GeLU Activation                                │
 │     --> Dense 2 (288 -> 144)                                                  │
 │  f) Add & LayerNorm 2: Norm(LayerNorm1_Output + FFN_Output)                   │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │ Output: (Batch, 72, 2)
                                         ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ 2. RESIDUAL CONV DECODER & UPSAMPLER                                         │
 │                                                                               │
 │  a) Reshape to 2D image: (B, 72, 2, 1)                                        │
 │  b) Conv2D (1 -> 2 filters, kernel 2x2): extracts local spatial features      │
 │  c) Residual Block: Conv2D -> ReLU -> Conv2D + BatchNorm                     │
 │  d) Dense Upsampling Layer (fc_upsample):                                     │
 │     --> Maps Pilot dimension 72 -> 1008 total grid elements (14 x 72)         │
 │  e) Conv2D Output Layer (2 -> 1 filter)                                       │
 │  f) Final Reshape: (B, 1008, 2) -> (B, 14, 72, 2)                             │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
                                         ▼
                   OUTPUT: Reconstructed Channel Grid (B, 14, 72, 2)
                           (14 Symbols x 72 Subcarriers x 2)
```

---

### 3. What is the role of Self-Attention & the Transformer?

- **Where is the Transformer?**
  The class `TransformerEncoderBlock` **is** the Transformer (specifically a Transformer Encoder).
- **Role of Self-Attention:**
  In wireless channels, physical subcarriers and symbols are strongly correlated due to **delay spread** (frequency selectivity) and **Doppler shift** (time selectivity).
  - The Self-Attention mechanism computes a $72 \times 72$ similarity matrix between all pilot elements.
  - It learns **the global frequency/time correlation** across all subcarriers and symbols.
  - Instead of treating pilots independently, attention allows each pilot position to "look at" all other pilots to denoise the LS estimates before reconstruction.

---

### 4. What does the Decoder do? Is it just reshaping back to `(14, 72, 2)`?

The Decoder does **much more than reshaping**:

1. **Feature Extraction:** `conv1` and the `Residual Block` extract non-linear 2D spatial features from the denoised pilot representations.
2. **Channel Interpolation & Extrapolation (Upsampling):**
   - The key layer is `fc_upsample` (a Dense/Linear layer).
   - It takes the **72 pilot features** and projects/interpolates them up to **1008 elements** ($14 \text{ OFDM symbols} \times 72 \text{ subcarriers}$).
3. **2D Grid Reconstruction:** Finally, it reshapes the 1008 elements into the **$(14 \text{ symbols}, 72 \text{ subcarriers}, 2 \text{ [Real/Imag]})$** 2D time-frequency channel matrix.

---

### Summary Table

| Stage | Input Shape | Output Shape | What it accomplishes |
|---|---|---|---|
| **Input** | `(B, 72, 2)` | `(B, 72, 2)` | Raw noisy LS estimates at 72 pilot positions |
| **`fc1` (Dense)** | `(B, 144)` | `(B, 432)` | Projects pilots into Query, Key, Value vectors |
| **Self-Attention** | $Q, K, V$ | `(B, 144)` | Learns global time-frequency correlations across pilots |
| **FFN (Dense)** | `(B, 144)` | `(B, 144)` | Non-linear feature refinement |
| **Conv2D + ResBlock** | `(B, 72, 2, 1)` | `(B, 72, 2, 2)` | Denoises 2D feature map |
| **`fc_upsample` (Dense)**| `(B, 72)` | `(B, 1008)` | **Upsamples from 72 pilots to 1008 full grid elements** |
| **Final Reshape** | `(B, 1008, 2)` | `(B, 14, 72, 2)` | **Reconstructs full grid (14 symbols $\times$ 72 subcarriers)** |