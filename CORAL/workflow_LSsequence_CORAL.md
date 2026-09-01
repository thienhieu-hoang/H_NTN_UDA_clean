Viewed note_HA02_Attention_Model.md:1-100
Viewed run_CORAL_LS_Attention.py:1-100

Based on the architecture of the **HA02 Attention Model** (Sparse LS Pilots $\rightarrow$ Transformer Encoder $\rightarrow$ Residual Conv Decoder $\rightarrow$ Dense $21\times$ Grid Upsampling), here is the recommended **Domain Adaptation approach** and the **exact layers to extract for CORAL**.

---

### 1. Recommended Domain Adaptation Approach

For Transformer / Attention-based channel estimation models, we recommend:

#### **Supervised Source Estimation + Unsupervised Multi-Layer CORAL (Direct UDA)**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(Y_{\text{src}}, \hat{X}_{\text{src}}) + \alpha \cdot \mathcal{L}_{\text{SSIM}} + \mathbf{\lambda_{\text{CORAL}} \cdot \mathcal{L}_{\text{CORAL}}(\mathbf{F}_{\text{src}}, \mathbf{F}_{\text{tgt}})}$$

* **Why this approach?**
  1. **High Training Stability:** Unlike convolutional GANs, Transformers with self-attention can suffer from gradient instabilities when coupled with a GAN discriminator. Direct CORAL domain alignment provides smooth, stable gradients directly to the attention weights and projection matrices.
  2. **Fast & Memory Efficient:** No need for extra discriminator forward/backward passes.
  3. **Directly Aligns Pilot Correlations:** Sparse pilot tone relationships are directly governed by the channel's Doppler and delay spreads—CORAL covariance matching directly corrects this shift.

---

### 2. Recommended Layers to Extract for CORAL

We recommend extracting features at **2 to 3 key transition points** in the HA02 dataflow:

```
[Input LS: 88x2]
       │
       ▼
┌───────────────────────────────────────┐
│       Transformer Encoder Block       │
│  (Multi-head self-attention on pilots)│
└───────────────────────────────────────┘
       │
       ├──► 🌟 [LAYER 1]: Encoder Output (Z_enc) ── Shape: [B, 176]   (PRIMARY RECOMMENDATION)
       ▼
┌───────────────────────────────────────┐
│     Residual Convolutional Block      │
│     (Local feature refinement)        │
└───────────────────────────────────────┘
       │
       ├──► 🌟 [LAYER 2]: Post-ResConv (Z_conv) ── Shape: [B, 352]   (SECONDARY)
       ▼
┌───────────────────────────────────────┐
│       Dense FC Upsample (88 -> 1848)  │
│       (Full Grid Latent Projection)   │
└───────────────────────────────────────┘
       │
       ├──► 🌟 [LAYER 3]: Latent Grid (Z_grid) ── Shape: [B, 4]      (OPTIONAL Global Pooled)
       ▼
[Output Grid: 132x14x2]
```

---

### Detailed Layer Breakdown:

| Layer Point | Variable / Block | Tensor Shape $[B, \dots]$ | Covariance Matrix Size | Why Extract Here? |
| :--- | :--- | :---: | :---: | :--- |
| **Layer 1 (Crucial)** | `TransformerEncoderBlock` output (after `LayerNorm 2` + FFN) | **`[B, 176]`** | **$[176, 176]$** | **Most Important Layer.** The self-attention block captures cross-pilot dependencies. Aligning covariance here forces the Transformer to learn **domain-invariant pilot correlation structures** regardless of user velocity or Doppler shift. |
| **Layer 2 (Recommended)** | `ResidualConvDecoderBlock` (after residual Conv2D + BatchNorm, before upsampling) | **`[B, 352]`** (`[B, 88, 2, 2]`) | **$[352, 352]$** | Aligns refined multi-channel pilot feature representations before the network expands the dimensions to the full grid. |
| **Layer 3 (Optional)** | Post-`fc_upsample` (Global channel pooled across the 1848 resource elements) | **`[B, 4]`** (or `[B, 64]` projected) | **$[4, 4]$** or **$[64, 64]$** | Aligns overall global energy distribution across the reconstructed time-frequency grid. |

---

### 3. Summary of How the CORAL Loss is Computed for HA02

1. **Extract Feature 1 (`enc_out` of shape `[B, 176]`):**
   * $\mathbf{C}_{\text{src}}^{(1)} = \frac{1}{B-1} (\mathbf{Z}_{\text{src}} - \bar{\mathbf{Z}}_{\text{src}})^T (\mathbf{Z}_{\text{src}} - \bar{\mathbf{Z}}_{\text{src}}) \in \mathbb{R}^{176 \times 176}$
   * $\mathbf{C}_{\text{tgt}}^{(1)} = \frac{1}{B-1} (\mathbf{Z}_{\text{tgt}} - \bar{\mathbf{Z}}_{\text{tgt}})^T (\mathbf{Z}_{\text{tgt}} - \bar{\mathbf{Z}}_{\text{tgt}}) \in \mathbb{R}^{176 \times 176}$
   * $\mathcal{L}_{\text{CORAL}}^{(1)} = \frac{1}{4 \cdot 176^2} \|\mathbf{C}_{\text{src}}^{(1)} - \mathbf{C}_{\text{tgt}}^{(1)}\|_F^2$

2. **Extract Feature 2 (`res_conv` of shape `[B, 352]`):**
   * $\mathcal{L}_{\text{CORAL}}^{(2)} = \frac{1}{4 \cdot 352^2} \|\mathbf{C}_{\text{src}}^{(2)} - \mathbf{C}_{\text{tgt}}^{(2)}\|_F^2$

3. **Total CORAL Loss:**
   $$\mathcal{L}_{\text{CORAL}} = \frac{1}{2} \left( \mathcal{L}_{\text{CORAL}}^{(1)} + \mathcal{L}_{\text{CORAL}}^{(2)} \right)$$

---

### Summary Recommendation:
* **Best Strategy:** Non-adversarial UDA with $\mathcal{L}_{\text{MSE}} + \lambda_{\text{CORAL}} \mathcal{L}_{\text{CORAL}}$ (with $\lambda_{\text{CORAL}} \approx 0.1\text{ to }0.5$).
* **Best Layers:** Extract `[Layer 1: Encoder Output (176-D)]` and `[Layer 2: Pre-Upsampling Residual Conv (352-D)]`. This yields small, well-conditioned covariance matrices ($176 \times 176$ and $352 \times 352$) that fit easily in memory and directly align the Transformer's attention mechanism.