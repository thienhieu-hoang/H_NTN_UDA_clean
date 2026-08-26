# Axial Self-Attention in OFDM Channel Estimation (OpenNTN)

## Overview & Context

In Non-Terrestrial Network (NTN) channel estimation, the channel matrix $\mathbf{H} \in \mathbb{C}^{N_{\text{subc}} \times N_{\text{symb}}}$ represents the complex channel response across $N_{\text{subc}} = 132$ subcarriers (frequency domain) and $N_{\text{symb}} = 14$ OFDM symbols (time domain). 

The script [`train_DnCNN_AxialAttention.py`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/train_DnCNN_AxialAttention.py) utilizes the model `DnCNN_ResNet_AxialAttention`, which incorporates the **`AxialAttention2D`** layer defined in [`utils_GAN.py`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/JMMD/helper/utils_GAN.py#L3830-L3912).

---

## What Does the Axial Attention Layer Do?

Standard 2D Self-Attention computes pairwise attention maps between every point $(h_1, w_1)$ and $(h_2, w_2)$ in a feature map of shape $[B, H, W, C]$. For a grid of size $132 \times 14 = 1848$ points, standard attention builds an $1848 \times 1848$ matrix (over 3.4 million elements per sample per head), which leads to quadratic computational and memory bottlenecks ($\mathcal{O}((HW)^2 C)$).

**Axial Attention** overcomes this limitation by factoring full 2D self-attention into two sequential 1D self-attentions along orthogonal axes:

1. **Subcarrier / Height Attention (Frequency Domain)**: Computes self-attention independently along the $H = 132$ subcarrier dimension for each of the $W = 14$ symbols.
2. **Symbol / Width Attention (Time Domain)**: Computes self-attention independently along the $W = 14$ symbol dimension for each of the $H = 132$ subcarriers.

```
Input Feature Map [B, H, W, C]
        │
        ▼
┌─────────────────────────────────────────┐
│ 1. Frequency (Height) Attention         │
│    - Reshape to [B * W, H, C_proj]      │
│    - Compute H x H Attention (132x132)  │
│    - Scale & Softmax                    │
│    - Residual connection with gamma_h   │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ 2. Time (Width) Attention               │
│    - Reshape to [B * H, W, C_proj]      │
│    - Compute W x W Attention (14x14)    │
│    - Scale & Softmax                    │
│    - Residual connection with gamma_w   │
└─────────────────────────────────────────┘
        │
        ▼
Output Feature Map [B, H, W, C]
```

### Mathematical & Algorithmic Breakdown

Given an input feature tensor $\mathbf{X} \in \mathbb{R}^{B \times H \times W \times C}$:

#### Step 1: Frequency (Subcarrier / Height) Attention
* **Query, Key, Value Projections** ($1 \times 1$ convolutions):
  $$\mathbf{Q}_h = \text{Conv2D}_{1\times1}(\mathbf{X}) \in \mathbb{R}^{B \times H \times W \times d_k}, \quad \mathbf{K}_h = \text{Conv2D}_{1\times1}(\mathbf{X}) \in \mathbb{R}^{B \times H \times W \times d_k}, \quad \mathbf{V}_h = \text{Conv2D}_{1\times1}(\mathbf{X}) \in \mathbb{R}^{B \times H \times W \times C}$$
  *(where $d_k = C / 8$)*.
* **Batch Rearrangement**: Treat symbol width $W$ as batch: $[B \cdot W, H, d_k]$.
* **Attention Map & Output**:
  $$\mathbf{A}_h = \text{Softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{H}}\right) \in \mathbb{R}^{(B \cdot W) \times H \times H}$$
  $$\mathbf{Y}_h = \mathbf{A}_h \mathbf{V}_h \in \mathbb{R}^{(B \cdot W) \times H \times C} \xrightarrow{\text{reshape/transpose}} \mathbb{R}^{B \times H \times W \times C}$$
* **Gated Residual Update**:
  $$\mathbf{X}' = \mathbf{X} + \gamma_h \cdot \mathbf{Y}_h$$

#### Step 2: Time (Symbol / Width) Attention
* **Query, Key, Value Projections** ($1 \times 1$ convolutions):
  $$\mathbf{Q}_w = \text{Conv2D}_{1\times1}(\mathbf{X}'), \quad \mathbf{K}_w = \text{Conv2D}_{1\times1}(\mathbf{X}'), \quad \mathbf{V}_w = \text{Conv2D}_{1\times1}(\mathbf{X}')$$
* **Batch Rearrangement**: Treat subcarrier height $H$ as batch: $[B \cdot H, W, d_k]$.
* **Attention Map & Output**:
  $$\mathbf{A}_w = \text{Softmax}\left(\frac{\mathbf{Q}_w \mathbf{K}_w^T}{\sqrt{W}}\right) \in \mathbb{R}^{(B \cdot H) \times W \times W}$$
  $$\mathbf{Y}_w = \mathbf{A}_w \mathbf{V}_w \in \mathbb{R}^{(B \cdot H) \times W \times C} \xrightarrow{\text{reshape}} \mathbb{R}^{B \times H \times W \times C}$$
* **Gated Residual Update**:
  $$\mathbf{X}'' = \mathbf{X}' + \gamma_w \cdot \mathbf{Y}_w$$

---

## How Is Axial Attention Helpful for Wireless Channel Estimation?

### 1. Physical Alignment with Wireless Propagation Characteristics
* **Frequency Axis (Subcarriers)**: Represents channel **Frequency Selectivity** caused by **Multipath Delay Spread** ($T_d$). Subcarrier self-attention captures correlation profiles across all subcarriers regardless of frequency distance.
* **Time Axis (OFDM Symbols)**: Represents channel **Time Selectivity** caused by **Doppler Frequency Shift** ($f_d$). Symbol self-attention captures time dynamics due to UE/satellite mobility (e.g. 20–30 m/s relative velocities in NTN).
* **Decoupled Physics**: Physical channels are mathematically separable into frequency response and temporal fading. Axial attention aligns directly with this physical separation instead of mixing time-frequency coordinates in an unstructured 2D matrix.

### 2. Significant Computational & Memory Savings (~12.65× Reduction)
For an OFDM resource grid of $H = 132$ subcarriers and $W = 14$ symbols ($HW = 1848$ Resource Elements):

| Parameter / Metric | Standard 2D Self-Attention | Axial Self-Attention | Improvement |
| :--- | :--- | :--- | :--- |
| **Complexity Formula** | $\mathcal{O}((HW)^2 C)$ | $\mathcal{O}(HW(H+W)C)$ | Factorized |
| **Pairwise Comparisons per Sample** | $1848^2 = 3,415,104$ | $14 \times 132^2 + 132 \times 14^2 = 269,808$ | **12.65× fewer operations** |
| **Attention Matrix Dimensions** | $[B, 1848, 1848]$ | $[B \cdot 14, 132, 132]$ & $[B \cdot 132, 14, 14]$ | Prevents GPU OOM |

### 3. Global Receptive Field without Deep Stacking
* Standard CNNs (e.g. 3×3 convolutions) only inspect immediate neighbors. To cover a 132-subcarrier span, traditional CNNs require dozens of layers or dilated kernels.
* Axial Attention provides an **instant global receptive field** across the entire subcarrier range ($132$) and symbol duration ($14$) within a single layer.

### 4. Stable Training with Zero-Initialized Gating ($\gamma_h, \gamma_w$)
* The scale factors $\gamma_h$ and $\gamma_w$ are initialized to $0.0$.
* At epoch 0, the model acts as a standard residual CNN (`out = x`).
* During training, the optimizer smoothly increases $\gamma_h$ and $\gamma_w$, gradually integrating global self-attention context without causing early training instability or gradient explosion.

---

## Summary Table

| Feature | Description |
| :--- | :--- |
| **Implementation** | `AxialAttention2D` in `utils_GAN.py` used by `DnCNN_ResNet_AxialAttention` |
| **Target Grid** | $132 \text{ subcarriers} \times 14 \text{ OFDM symbols} \times 2 \text{ channels (Real/Imag)}$ |
| **Core Operation** | Factored 1D self-attention along subcarriers ($H$), followed by 1D self-attention along symbols ($W$) |
| **Key Advantage** | Captures delay spread & Doppler dynamics with 12.65× lower compute & memory overhead |
