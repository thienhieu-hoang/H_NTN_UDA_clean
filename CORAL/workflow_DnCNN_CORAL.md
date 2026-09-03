# Domain Adaptation Workflow for DnCNN Model (OpenNTN)

This document describes the **Unsupervised Domain Adaptation (UDA)** workflow, architectural principles, and mathematical formulations for the **DnCNN (CNNGenerator)** deep learning models applied to 5G Non-Terrestrial Network (NTN) channel estimation, covering:
1. **Direct Multi-Layer CORAL Alignment**: [`train_CORAL_DnCNN.py`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/CORAL/train_CORAL_DnCNN.py)
2. **Projection-Head CORAL Alignment**: [`train_CORALpHead_DnCNN.py`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/CORAL/train_CORALpHead_DnCNN.py)

---

## 1. DnCNN (CNNGenerator) Architecture Overview

The DnCNN backbone takes a 2D time-frequency channel grid (e.g. Linear Interpolation $\mathbf{H}_{\text{LI}}$, Practical Interpolation $\mathbf{H}_{\text{Prac}}$, or Least Squares $\mathbf{H}_{\text{LS}}$) and denoises/reconstructs the full perfect complex channel grid:

$$\mathbf{H}_{\text{in}} \in \mathbb{R}^{132 \times 14 \times 2} \xrightarrow{\quad \text{DnCNN} \quad} \hat{\mathbf{H}}_{\text{out}} \in \mathbb{R}^{132 \times 14 \times 2}$$

### 4-SameShapeBlock Architecture with Projection-Head CORAL

The diagram below illustrates the default **4-SameShapeBlock** network topology (`N_BLOCKS = 4`, configurable to 6 blocks via `--n-blocks 6`) coupled with multi-layer feature extraction and projection-head alignment:
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
       ├──► 🌟 [block_2]: Spatial Feature Map [B, 132, 14, 128]  (PRIMARY CORAL POINT)
       ▼
┌────────────────────────────────────────────────────────┐
│ Block 3: SameShapeBlock (Conv2D -> INorm -> LReLU)     │ ──► [B, 132, 14, 128]
└────────────────────────────────────────────────────────┘
       │
       ├──► 🌟 [block_3]: Spatial Feature Map [B, 132, 14, 128]  (SECONDARY CORAL POINT)
       ▼
┌────────────────────────────────────────────────────────┐
│ Block 4: SameShapeBlock (Conv2D -> INorm -> LReLU)     │ ──► [B, 132, 14, 64]
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
* **$N_{\text{blocks}} = 4$**:
  * Block 1: $64$ channels
  * Block 2: $128$ channels
  * Block 3: $128$ channels
  * Block 4: $64$ channels
* **$N_{\text{blocks}} = 6$**:
  * Block 1: $64$ channels
  * Block 2: $128$ channels
  * Block 3: $512$ channels
  * Block 4: $512$ channels
  * Block 5: $128$ channels
  * Block 6: $64$ channels

---

## 3. Spatial Feature Pooling for 4D CNN Representations

### The High-Dimensionality Challenge:
Each SameShapeBlock produces a 4D tensor $\mathbf{Z} \in \mathbb{R}^{B \times 132 \times 14 \times C_k}$.
* Direct spatial flattening produces vectors of dimension $D = 132 \times 14 \times C_k = 1848 \times C_k$.
* For $C_k = 128$, $D = 236,544$. Computing a sample covariance matrix of size $236,544 \times 236,544$ would require over **220 GB of VRAM**, crashing the GPU.

### Spatial Global Average Pooling (GAP) Solution:
To perform second-order domain alignment efficiently and robustly, spatial Global Average Pooling is applied across the time-frequency subcarrier and OFDM symbol dimensions:

$$\bar{\mathbf{z}}_{b, c} = \frac{1}{H \times W} \sum_{h=1}^{H=132} \sum_{w=1}^{W=14} \mathbf{Z}_{b, h, w, c} \quad \implies \quad \bar{\mathbf{Z}} \in \mathbb{R}^{B \times C_k}$$

```
====================================================================================================
SPATIAL GLOBAL AVERAGE POOLING (GAP)
====================================================================================================
   Z_block [B, 132, 14, C_k] ──► Spatial GAP (Mean over 132x14) ──► Z_pooled [B, C_k]
====================================================================================================
```

### Physical & Domain Adaptation Rationale:
1. **Time-Frequency Translation Invariance**: In 5G NTN channels, Doppler frequency shifts and propagation delays cause phase rotations and grid translations. Global spatial pooling captures the **energy and cross-channel correlation profile** while remaining invariant to spatial shifts.
2. **Compact & Well-Conditioned Covariance**:
   * For $C_k = 128$: Covariance matrix is $[128 \times 128]$ (only 16,384 elements).
   * For $C_k = 512$: Covariance matrix is $[512 \times 512]$ (262,144 elements).
3. **High GPU Training Speed**: Enables fast training on standard GPUs with `@tf.function` compiled graph execution.

---

## 4. Method 1: Direct CORAL Alignment (`train_CORAL_DnCNN.py`)

In [`train_CORAL_DnCNN.py`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/CORAL/train_CORAL_DnCNN.py), the domain covariance alignment is applied directly to the pooled feature vectors $\bar{\mathbf{Z}}^{(k)}$:

```
[Source Batch] ──► DnCNN ──► Z_src [B, 132, 14, C_k] ──► Spatial GAP ──► Z_pool_src [B, C_k] ──┐
                                                                                               ├──► CORAL Loss
[Target Batch] ──► DnCNN ──► Z_tgt [B, 132, 14, C_k] ──► Spatial GAP ──► Z_pool_tgt [B, C_k] ──┘
```

$$\mathcal{L}_{\text{CORAL}} = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \frac{1}{4 C_k^2} \|\mathbf{C}(\bar{\mathbf{Z}}_{\text{src}}^{(k)}) - \mathbf{C}(\bar{\mathbf{Z}}_{\text{tgt}}^{(k)})\|_F^2$$

---

## 5. Method 2: Dedicated Projection-Head Network (`train_CORALpHead_DnCNN.py`)

In [`train_CORALpHead_DnCNN.py`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/CORAL/train_CORALpHead_DnCNN.py), each selected intermediate block is connected to a **Dedicated Non-Linear Projection Head**:

```
[Intermediate Block Feature: [B, 132, 14, C_k]]
               │
               ▼  (Spatial Global Average Pooling)
        [B, C_k]  (Raw Feature Z)
               │
               ▼
┌────────────────────────────────────────────────────────┐
│ Projection Head Network (Trained Jointly during UDA)   │
│ 1. Dense(hidden_dim = max(C_k/2, 128))                │
│ 2. LayerNormalization(eps=1e-5)                        │
│ 3. GeLU Non-Linear Activation                          │
│ 4. Dense(proj_dim = 64)                                │
└────────────────────────────────────────────────────────┘
               │
               ▼
     [B, d_proj = 64] (Projected Feature Embedding P)
               │
               ▼
   [Projected CORAL Covariance Alignment]
```

### Projection Head Layer Configurations:

| Extraction Layer | $N_{\text{blocks}}=4$ Channels ($C_k$) | $N_{\text{blocks}}=6$ Channels ($C_k$) | Projection Head MLP Architecture | Output Embedding Dim ($d_{\text{proj}}$) |
| :--- | :---: | :---: | :--- | :---: |
| **`block_1`** | $64$ | $64$ | $[64 \to 128 \to 64]$ | $64$ |
| **`block_2`** | $128$ | $128$ | $[128 \to 128 \to 64]$ | $64$ |
| **`block_3`** | $128$ | $512$ | $[128 \to 128 \to 64]$ (4b) / $[512 \to 256 \to 64]$ (6b) | $64$ |
| **`block_4`** | $64$ | $512$ | $[64 \to 128 \to 64]$ (4b) / $[512 \to 256 \to 64]$ (6b) | $64$ |

### Advantages of Non-Linear Projection Heads:
1. **Dimension Uniformity**: Projects disparate feature channel widths ($C_2=128, C_3=512$) into equal-sized $d_{\text{proj}}=64$ latent subspaces for balanced multi-layer domain alignment.
2. **Non-Linear Manifold Alignment**: LayerNorm + GeLU non-linear projection aligns higher-order structural manifold properties beyond simple linear statistics.
3. **Decoupled Task & Domain Representations**: Allows the DnCNN backbone to retain channel estimation reconstruction fidelity while the projection head learns domain-invariant subspaces.
4. **Zero Inference Overhead**: The projection heads are used only during backpropagation for domain adaptation loss calculation and are discarded during inference.
5. **Dual Feature Checkpointing**: `extracted_features.mat` captures both `raw` (pre-projection $[B, C_k]$) and `proj` (post-projection $[B, 64]$) features across `begin`, `mid`, and `last` epochs.

---

## 6. Mathematical Formulation & Dynamic Loss Annealing

### 1. Spatial Sample Covariance Matrix
For a batch $\mathbf{X} \in \mathbb{R}^{B \times d}$ (where $\mathbf{X} = \bar{\mathbf{Z}}$ for direct CORAL, or $\mathbf{X} = \mathbf{P}$ for projection-head CORAL) with batch mean $\bar{\mathbf{x}} = \frac{1}{B} \sum_{i=1}^B \mathbf{x}_i$:

$$\mathbf{C}(\mathbf{X}) = \frac{1}{B - 1} (\mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^T)^T (\mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^T) \in \mathbb{R}^{d \times d}$$

### 2. Multi-Head Projected CORAL Loss
For selected layers $\mathcal{K} = \{\text{block\_2}, \text{block\_3}\}$:

$$\mathcal{L}_{\text{CORAL}} = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \frac{1}{4 d_{\text{proj}}^2} \|\mathbf{C}(\mathbf{P}_{\text{src}}^{(k)}) - \mathbf{C}(\mathbf{P}_{\text{tgt}}^{(k)})\|_F^2$$

### 3. Dynamic SSIM-to-MSE Weight Annealing
During early epochs, structural similarity (SSIM) guides the network to capture global channel envelope geometry. In later epochs, MSE focuses on precise complex-valued phase and amplitude fitting:

$$\alpha(e) = \alpha_{\text{start}} - \frac{e}{E - 1} (\alpha_{\text{start}} - \alpha_{\text{end}}), \quad e \in \{0, 1, \dots, E-1\}$$

where $\alpha_{\text{start}} = 0.95$ and $\alpha_{\text{end}} = 0.05$.

$$\mathcal{L}_{\text{est}}(Y_{\text{src}}, \hat{X}_{\text{src}}) = (1 - \alpha(e)) \cdot \mathcal{L}_{\text{MSE}}(Y_{\text{src}}, \hat{X}_{\text{src}}) + \alpha(e) \cdot \left(1 - \text{SSIM}(Y_{\text{src}}, \hat{X}_{\text{src}})\right)$$

### 4. Joint Total Optimization Objective

$$\min_{\Theta_{\text{DnCNN}}, \{\Theta_{\text{Head}}^{(k)}\}} \mathcal{L}_{\text{est}}(Y_{\text{src}}, \hat{X}_{\text{src}}) + \lambda_{\text{CORAL}} \cdot \mathcal{L}_{\text{CORAL}}$$

where $\lambda_{\text{CORAL}} = 0.5$ (default).

---

## 7. Comparison: Direct CORAL vs Projection-Head CORAL

| Feature | Direct CORAL (`train_CORAL_DnCNN.py`) | Projection-Head CORAL (`train_CORALpHead_DnCNN.py`) |
| :--- | :--- | :--- |
| **Alignment Target** | Raw pooled features $\bar{\mathbf{Z}} \in \mathbb{R}^{B \times C_k}$ | Non-linear embeddings $\mathbf{P} \in \mathbb{R}^{B \times 64}$ |
| **Subspace Mapping** | Linear identity | 2-layer MLP (Dense $\to$ LayerNorm $\to$ GeLU $\to$ Dense) |
| **Covariance Dimensions** | Varies by layer ($128 \times 128$, $512 \times 512$) | Fixed $64 \times 64$ for all layers |
| **Cross-Layer Weighting** | Normalized by $1/(4 C_k^2)$ | Balanced uniformly across equal $d_{\text{proj}} = 64$ |
| **Extracted Features** | `raw` GAP features $[N, C_k]$ | `raw` $[N, C_k]$ **and** `proj` $[N, 64]$ |
| **Test-Time Complexity** | Zero overhead | Zero overhead (heads detached at inference) |

---

## 8. Execution Commands & Generated Artifacts

### Running Experiments
```bash
# 1. Direct CORAL UDA on DnCNN (4 residual blocks)
python train_CORAL_DnCNN.py --snr 5 --coral-layers block_2 block_3 --n-blocks 4 --domain-weight 0.5 --save-features

# 2. Projection-Head CORAL UDA on DnCNN (4 residual blocks, proj_dim = 64)
python train_CORALpHead_DnCNN.py --snr 5 --coral-layers block_2 block_3 --n-blocks 4 --proj-dim 64 --domain-weight 0.5 --save-features

# 3. Direct CORAL UDA on DnCNN (6 residual blocks)
python train_CORAL_DnCNN.py --snr 5 --coral-layers block_2 block_3 --n-blocks 6 --domain-weight 0.5 --save-features

# 4. Source-Only Baseline (no domain adaptation)
python train_CORAL_DnCNN.py --snr 5 --only-source

# 5. Quick Sanity Code Test (5 epochs on subset)
python train_CORALpHead_DnCNN.py --test-code --snr 5 --save-features
```

### Generated Artifacts in `results/`
* **`testChannel_source.mat` & `testChannel_target.mat`**: Held-out test channels for MATLAB BER simulations (including ground truth, input, LI benchmark, and model output).
* **`sample_reconstructions.mat`**: Exact channel grid frames across 4 splits (Source Train, Source Test, Target Train, Target Test) for MATLAB replotting.
* **`training_history.mat`**: Numerical progression across all epochs (Total Loss, Est Loss, CORAL Loss, 4-way NMSE dB, MSE, and SSIM).
* **`evaluation_results.mat`**: Summary scalar metrics on test splits.
* **`final_epoch.txt`**: Consolidated text evaluation report.
* **`extracted_features.mat`**: Raw and projected activations captured at `begin`, `mid`, and `last` training checkpoints with `train_indices_src` and `train_indices_tgt`.
* **PDF Figures**:
  * `loss_total.pdf`: Total Loss, Estimation Loss, and CORAL Loss progression.
  * `metrics_nmse_db.pdf`: 4-way NMSE (dB) curves.
  * `metrics_mse.pdf`: 4-way MSE curves.
  * `metrics_ssim.pdf`: 4-way SSIM curves.
  * `metrics_summary_2x2.pdf`: Consolidated 2x2 multi-panel layout.
  * `recon_source_train.pdf`, `recon_source_test.pdf`, `recon_target_train.pdf`, `recon_target_test.pdf`: 1x3 channel reconstruction heatmaps.
