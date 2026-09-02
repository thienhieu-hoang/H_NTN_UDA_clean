# Domain Adaptation Workflow for DnCNN Model (OpenNTN)

This document describes the **Unsupervised Domain Adaptation (UDA)** workflow and architectural principles for the **DnCNN (CNNGenerator)** deep learning model applied to 5G Non-Terrestrial Network (NTN) channel estimation, as implemented in [`train_CORAL_DnCNN.py`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/CORAL/train_CORAL_DnCNN.py).

---

## 1. DnCNN (CNNGenerator) Architecture Overview

The DnCNN model takes a 2D time-frequency channel grid (e.g. Linear Interpolation $\mathbf{H}_{\text{LI}}$, Practical Interpolation $\mathbf{H}_{\text{Prac}}$, or Least Squares $\mathbf{H}_{\text{LS}}$) and denoises/reconstructs the full perfect channel grid:

$$\mathbf{H}_{\text{in}} \in \mathbb{R}^{132 \times 14 \times 2} \xrightarrow{\quad \text{DnCNN} \quad} \hat{\mathbf{H}}_{\text{out}} \in \mathbb{R}^{132 \times 14 \times 2}$$

```
[Input Channel Grid: 132x14x2 (Real, Imag)]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Block 1: SameShapeBlock (Conv2D -> INorm -> LReLU)     │ ──► [B, 132, 14, 64]
└────────────────────────────────────────────────────────┘
       │
       ├──► 🌟 [block_1]: Spatial Feature Map [B, 132, 14, 64]
       ▼
┌────────────────────────────────────────────────────────┐
│ Block 2: SameShapeBlock (Conv2D -> INorm -> LReLU)     │ ──► [B, 132, 14, 128]
└────────────────────────────────────────────────────────┘
       │
       ├──► 🌟 [block_2]: Spatial Feature Map [B, 132, 14, 128]  (DEFAULT CORAL POINT)
       ▼
┌────────────────────────────────────────────────────────┐
│ Block 3: SameShapeBlock (Conv2D -> INorm -> LReLU)     │ ──► [B, 132, 14, 512]
└────────────────────────────────────────────────────────┘
       │
       ├──► 🌟 [block_3]: Spatial Feature Map [B, 132, 14, 512]  (DEFAULT CORAL POINT)
       ▼
┌────────────────────────────────────────────────────────┐
│ Block 4: SameShapeBlock (Conv2D -> INorm -> LReLU)     │ ──► [B, 132, 14, 512]
└────────────────────────────────────────────────────────┘
       │
       ├──► 🌟 [block_4]: Spatial Feature Map [B, 132, 14, 512]
       ▼
┌────────────────────────────────────────────────────────┐
│ Block 5: SameShapeBlock (Conv2D -> INorm -> LReLU)     │ ──► [B, 132, 14, 128]
└────────────────────────────────────────────────────────┘
       │
       ├──► 🌟 [block_5]: Spatial Feature Map [B, 132, 14, 128]
       ▼
┌────────────────────────────────────────────────────────┐
│ Block 6: SameShapeBlock (Conv2D -> INorm -> LReLU)     │ ──► [B, 132, 14, 64]
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Output Conv2D (1x1 or 3x3) -> Tanh (2 channels)        │
└────────────────────────────────────────────────────────┘
       │
       ▼
[Output Channel Grid: 132x14x2]
```

---

## 2. Channel Pyramid Filter Formula

The number of feature channels at each block $i \in \{0, 1, \dots, N_{\text{blocks}}-1\}$ is governed by a symmetric pyramid scaling function:

$$\text{filters}(i) = \begin{cases} 
64, & \text{if } i = 0 \text{ or } i = N_{\text{blocks}} - 1 \\
\min(32 \times 4^i, 1024), & \text{if } i < \lfloor N_{\text{blocks}} / 2 \rfloor \\
\min(32 \times 4^{N_{\text{blocks}} - 1 - i}, 1024), & \text{if } i \ge \lfloor N_{\text{blocks}} / 2 \rfloor 
\end{cases}$$

### Filter Counts for Standard Configurations:
* **$N_{\text{blocks}} = 4$**: $[64, 128, 128, 64]$
* **$N_{\text{blocks}} = 6$ (Default)**: $[64, 128, 512, 512, 128, 64]$

---

## 3. Spatial Feature Pooling for CORAL Alignment

### The High-Dimensionality Challenge in 4D CNNs:
Each block output is a 4D tensor $\mathbf{Z} \in \mathbb{R}^{B \times 132 \times 14 \times C_k}$.
* Direct spatial flattening produces vectors of dimension $D = 132 \times 14 \times C_k = 1848 \times C_k$.
* For $C_k = 128$, $D = 236,544$. A $236,544 \times 236,544$ covariance matrix would require over **220 GB of VRAM**, crashing the GPU.

### Spatial Global Average Pooling (GAP) Solution:
To perform second-order domain alignment efficiently and robustly, spatial Global Average Pooling is applied across the time-frequency subcarrier and OFDM symbol dimensions:

$$\bar{\mathbf{z}}_{b, c} = \frac{1}{H \times W} \sum_{h=1}^{H=132} \sum_{w=1}^{W=14} \mathbf{Z}_{b, h, w, c} \quad \implies \quad \bar{\mathbf{Z}} \in \mathbb{R}^{B \times C_k}$$

```
====================================================================================================
SPATIAL GLOBAL AVERAGE POOLING (GAP) CORAL ALIGNMENT
====================================================================================================
   Z_block [B, 132, 14, C_k] ──► Spatial GAP (Mean over 132x14) ──► Z_pooled [B, C_k]
                                                                          │
                                                                          ▼
                                                               Covariance Matrix [C_k x C_k]
                                                                          │
                                                                          ▼
                                                               CORAL Loss: ||Cov_src - Cov_tgt||_F^2
====================================================================================================
```

### Advantages of Spatial GAP in CORAL:
1. **Physical Channel Invariance**: In 5G NTN, Doppler shifts and delay spreads introduce shifts along time and frequency axes. Global spatial pooling captures the **energy and correlation profile across feature channels** while remaining invariant to absolute spatial translation.
2. **Compact & Well-Conditioned Covariance**:
   * For `block_2` ($C_k = 128$): Covariance matrix is $[128 \times 128]$.
   * For `block_3` ($C_k = 512$): Covariance matrix is $[512 \times 512]$.
3. **High Throughput & Low Memory**: Enables fast training on standard GPUs (e.g. RTX 3080/4090) with compiled `@tf.function` execution.

---

## 4. Multi-Layer Extraction Points

| Layer Key | Extraction Block | Spatial Tensor Shape | Pooled Vector Shape | Physical & Domain Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **`block_1`** | Block 1 Output | `[B, 132, 14, 64]` | `[B, 64]` | Low-level time-frequency edge & pilot boundary features. |
| **`block_2`** | Block 2 Output | `[B, 132, 14, 128]` | `[B, 128]` | **Default Primary Point.** Captures local subcarrier delay-correlation patterns across neighboring subcarriers. |
| **`block_3`** | Block 3 Output | `[B, 132, 14, 512]` | `[B, 512]` | **Default Secondary Point.** Deep latent representations capturing rich non-linear Doppler-delay couplings. |
| **`block_4`** | Block 4 Output | `[B, 132, 14, 512]` | `[B, 512]` | Symmetric decoding block prior to channel resolution refinement. |
| **`block_5`** | Block 5 Output | `[B, 132, 14, 128]` | `[B, 128]` | Multi-scale channel feature reconstruction. |
| **`block_6`** | Block 6 Output | `[B, 132, 14, 64]` | `[B, 64]` | Final refinement stage before channel grid synthesis. |

---

## 5. Mathematical Formulation & Dynamic Loss Annealing

### 1. Spatial Covariance Matrix
For a globally pooled batch $\mathbf{Z} \in \mathbb{R}^{B \times C_k}$ with batch mean $\bar{\mathbf{z}} = \frac{1}{B} \sum_{i=1}^B \mathbf{z}_i$:

$$\mathbf{C}(\mathbf{Z}) = \frac{1}{B - 1} (\mathbf{Z} - \mathbf{1}\bar{\mathbf{z}}^T)^T (\mathbf{Z} - \mathbf{1}\bar{\mathbf{z}}^T) \in \mathbb{R}^{C_k \times C_k}$$

### 2. Multi-Layer CORAL Loss
For a set of selected layers $\mathcal{K}$ (e.g., `['block_2', 'block_3']`):

$$\mathcal{L}_{\text{CORAL}} = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \frac{1}{4 C_k^2} \|\mathbf{C}(\mathbf{Z}_{\text{src}}^{(k)}) - \mathbf{C}(\mathbf{Z}_{\text{tgt}}^{(k)})\|_F^2$$

### 3. Dynamic SSIM-to-MSE Weight Annealing
During the initial epochs, structural similarity (SSIM) guides the network to capture global channel envelope geometry. In later epochs, MSE focuses on precise complex-valued phase and amplitude fitting:

$$\alpha(e) = \alpha_{\text{start}} - \frac{e}{E - 1} (\alpha_{\text{start}} - \alpha_{\text{end}}), \quad e \in \{0, 1, \dots, E-1\}$$

where $\alpha_{\text{start}} = 0.95$ and $\alpha_{\text{end}} = 0.05$.

$$\mathcal{L}_{\text{est}}(Y_{\text{src}}, \hat{X}_{\text{src}}) = (1 - \alpha(e)) \cdot \mathcal{L}_{\text{MSE}}(Y_{\text{src}}, \hat{X}_{\text{src}}) + \alpha(e) \cdot \left(1 - \text{SSIM}(Y_{\text{src}}, \hat{X}_{\text{src}})\right)$$

### 4. Joint Total Optimization Objective

$$\min_{\Theta_{\text{DnCNN}}} \mathcal{L}_{\text{est}}(Y_{\text{src}}, \hat{X}_{\text{src}}) + \lambda_{\text{CORAL}} \cdot \mathcal{L}_{\text{CORAL}}$$

where $\lambda_{\text{CORAL}} = 0.5$ (default).

---

## 6. Execution Commands & Generated Artifacts

### Running Experiments
```bash
# Multi-layer CORAL UDA on DnCNN (block_2 + block_3, 6 residual blocks, SNR = 5 dB)
python train_CORAL_DnCNN.py --snr 5 --coral-layers block_2 block_3 --n-blocks 6 --domain-weight 0.5 --save-features

# Source-only baseline (no domain adaptation)
python train_CORAL_DnCNN.py --snr 5 --only-source

# Quick code test (subset of data)
python train_CORAL_DnCNN.py --test-code --snr 5 --save-features
```

### Generated Artifacts in `results/`
* **`testChannel_source.mat` & `testChannel_target.mat`**: Held-out test channels for MATLAB BER simulations (including ground truth, input, LI benchmark, and model output).
* **`sample_reconstructions.mat`**: Exact channel grid frames across 4 splits (Source Train, Source Test, Target Train, Target Test) for MATLAB replotting.
* **`training_history.mat`**: Numerical progression across all epochs (Total Loss, Est Loss, CORAL Loss, 4-way NMSE dB, MSE, and SSIM).
* **`evaluation_results.mat`**: Summary scalar metrics on test splits.
* **`final_epoch.txt`**: Consolidated text evaluation report.
* **`extracted_features.mat`**: Feature activations captured at `begin`, `mid`, and `last` training checkpoints.
* **PDF Figures**:
  * `loss_total.pdf`: Total Loss, Estimation Loss, and CORAL Loss progression.
  * `metrics_nmse_db.pdf`: 4-way NMSE (dB) curves.
  * `metrics_mse.pdf`: 4-way MSE curves.
  * `metrics_ssim.pdf`: 4-way SSIM curves.
  * `metrics_summary_2x2.pdf`: Consolidated 2x2 multi-panel layout.
  * `recon_source_train.pdf`, `recon_source_test.pdf`, `recon_target_train.pdf`, `recon_target_test.pdf`: 1x3 channel reconstruction heatmaps.
