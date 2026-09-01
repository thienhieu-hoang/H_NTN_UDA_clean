# Domain Adaptation Workflow for HA02 Attention Model (OpenNTN)

This document describes the **Unsupervised Domain Adaptation (UDA)** strategies for the **HA02 Transformer-Convolutional Attention Model** for 5G Non-Terrestrial Network (NTN) channel estimation, comparing **Direct CORAL Alignment** with **Non-Linear Projection Head Alignment**.

---

## 1. HA02 Attention Model Architecture Overview

The HA02 model processes sparse pilot channel observations and upsamples them to the full time-frequency grid:
$$\text{Sparse Pilots } \mathbf{H}_{\text{LS}} \in \mathbb{C}^{88} \xrightarrow{\quad \text{HA02} \quad} \hat{\mathbf{H}}_{\text{grid}} \in \mathbb{C}^{132 \times 14}$$

```
[Input LS Pilots: 88x2]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│               Transformer Encoder Block                │
│  (Multi-Head Self-Attention on pilots + LN1 + FFN + LN2│
└────────────────────────────────────────────────────────┘
       │
       ├──► 🌟 [LAYER 1]: Encoder Output (Z_enc) ── Shape: [B, 176]   (PRIMARY EXTRACTION POINT)
       ▼
┌────────────────────────────────────────────────────────┐
│             Residual Convolutional Decoder             │
│        (Conv2D -> ReLU -> Conv2D + BatchNorm)          │
└────────────────────────────────────────────────────────┘
       │
       ├──► 🌟 [LAYER 2]: Post-ResConv (Z_conv) ── Shape: [B, 352]   (SECONDARY EXTRACTION POINT)
       ▼
┌────────────────────────────────────────────────────────┐
│             Dense FC Upsampler (88 -> 1848)            │
│             (Spatial Pilot -> Grid Expansion)          │
└────────────────────────────────────────────────────────┘
       │
       ├──► 🌟 [LAYER 3]: Latent Grid (Z_grid) ── Shape: [B, 2]      (OPTIONAL GLOBAL POOLED)
       ▼
[Output Grid: 132x14x2]
```

---

## 2. Multi-Layer Feature Extraction Points

| Layer Name | Block Output | Raw Tensor Shape | Flattened / Pooled Dimension | Physical & Domain Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **`layer1`** | `TransformerEncoderBlock` output (after LN2) | `[B, 88, 2]` | **`[B, 176]`** | **Highest Potential.** Self-attention captures cross-pilot correlations. In NTN, Doppler shift alters pilot correlation phase across symbols. Aligning here forces the attention heads to learn **Doppler-invariant correlation patterns**. |
| **`layer2`** | `ResidualConvDecoderBlock` (pre-upsampling) | `[B, 88, 2, 2]` | **`[B, 352]`** | Refined multi-channel local pilot features before spatial dimension expansion. |
| **`layer3`** | Post-Upsample Global Pooled | `[B, 1848, 2, 2]` | **`[B, 2]`** | Global average energy distribution across subcarriers and symbols. |

---

## 3. Direct CORAL Alignment vs. Projection Head Alignment

```
====================================================================================================
A) DIRECT CORAL ALIGNMENT (Standard)
====================================================================================================
   Z_enc [B, 176] ───────────────────────────► Covariance Matrix [176 x 176]
                                                    │
                                                    ▼
                                           CORAL Loss: ||Cov_src - Cov_tgt||_F^2

====================================================================================================
B) NON-LINEAR PROJECTION HEAD ALIGNMENT (Recommended)
====================================================================================================
   Z_enc [B, 176] ──► [Main Channel Estimator Branch] ──► Residual Decoder ──► Full Grid [132x14x2]
         │
         └──► [Domain Adaptation Branch] (Active only in Training)
                    │
                    ▼
              ┌─────────────────────────────────────────┐
              │     Dedicated Projection Head (MLP)     │
              │  Dense(128) -> LayerNorm -> GeLU        │
              │  -> Dense(64)                           │
              └───────────────────┬─────────────────────┘
                                  │
                                  ▼
                              P [B, 64] ──► Covariance Matrix [64 x 64]
                                                │
                                                ▼
                                      Projected CORAL Loss: ||Cov_src(P) - Cov_tgt(P)||_F^2
====================================================================================================
```

### Why Use a Projection Head?

1. **Information Bottleneck & Capacity Preservation:**
   * Direct CORAL on $Z_{\text{enc}}$ forces the raw representation to have identical covariance across domains, which can force the network to discard fine-grained pilot magnitude details useful for channel reconstruction.
   * With a non-linear projection head $g_{\text{proj}}(Z) \to P$, the main representation $Z_{\text{enc}}$ retains full signal richness for the decoder, while $P$ isolates the domain-invariant subspace for covariance alignment.

2. **Well-Conditioned Covariance with Small Batches ($B \ll d$):**
   * With training batch sizes like $B = 16$ or $B = 32$, estimating a $176 \times 176$ or $352 \times 352$ covariance matrix is mathematically rank-deficient and noisy.
   * Projecting down to **$d_{\text{proj}} = 64$** produces a compact, well-conditioned $64 \times 64$ covariance matrix that yields cleaner, more stable gradients.

3. **Non-Linear Metric Alignment (Kernelized Covariance):**
   * Standard CORAL matches linear second-order moments ($\mathbb{E}[Z Z^T]$).
   * Passing features through GeLU non-linearities transforms linear covariance matching into **non-linear kernel alignment**, capturing higher-order domain discrepancies.

4. **Zero Inference Cost:**
   * All projection heads are completely detached during testing/deployment. Inference latency and memory remain 100% identical to the base HA02 model.

---

## 4. Projection Head Network Architecture

Each extracted layer has a **dedicated, custom-sized non-linear projection head network**:

```
                       ┌────────────────────────────────────────┐
                       │   Input Feature: Z_layer [B, d_in]     │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  Linear Projection: Dense(hidden_dim)  │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │     Layer Normalization (eps=1e-5)     │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │   Non-Linear Activation: GeLU          │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  Output Projection: Dense(proj_dim)    │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │   Projected Latent: P_layer [B, 64]    │
                       └────────────────────────────────────────┘
```

### Layer-by-Layer Projection Head Dimensions

| Extracted Layer | Input Tensor | Input Dim ($d_{\text{in}}$) | Hidden Dim | Projected Dim ($d_{\text{proj}}$) | Output Covariance Size |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`layer1`** (Transformer) | $Z_{\text{enc}}$ | **176** | 128 | **64** | $[64 \times 64]$ |
| **`layer2`** (ResConv) | $Z_{\text{conv}}$ | **352** | 128 | **64** | $[64 \times 64]$ |
| **`layer3`** (Global Grid) | $Z_{\text{grid}}$ | **2** | 32 | **32** | $[32 \times 32]$ |

---

## 5. Adaptive Multi-Head Manager

The training pipeline in [`run_CORALpHead_LS_Attention.py`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/CORAL/run_CORALpHead_LS_Attention.py) dynamically instantiates and coordinates projection head networks based on the `--coral-layers` command-line argument:

### Case 1: Single Layer (`--coral-layers layer1`)
* Instantiates **1 projection head**:
  $$\text{ProjHead}_{\text{layer1}}: [176 \to 128 \to 64]$$
* Total Loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(Y_{\text{src}}, \hat{X}_{\text{src}}) + \alpha \mathcal{L}_{\text{SSIM}} + \lambda_{\text{CORAL}} \cdot \mathcal{L}_{\text{CORAL}}(\mathbf{P}_{\text{layer1}}^{\text{src}}, \mathbf{P}_{\text{layer1}}^{\text{tgt}})$$

### Case 2: Multi-Layer (`--coral-layers layer1 layer2`)
* Instantiates **2 distinct projection heads**:
  $$\text{ProjHead}_{\text{layer1}}: [176 \to 128 \to 64] \quad (\text{Transformer Pilot Attention Subspace})$$
  $$\text{ProjHead}_{\text{layer2}}: [352 \to 128 \to 64] \quad (\text{Residual Conv Spatial Subspace})$$
* Total Projected CORAL Loss:
  $$\mathcal{L}_{\text{CORAL}} = \frac{1}{2}\left( \mathcal{L}_{\text{CORAL}}(\mathbf{P}_{\text{layer1}}^{\text{src}}, \mathbf{P}_{\text{layer1}}^{\text{tgt}}) + \mathcal{L}_{\text{CORAL}}(\mathbf{P}_{\text{layer2}}^{\text{src}}, \mathbf{P}_{\text{layer2}}^{\text{tgt}}) \right)$$

### Case 3: All 3 Layers (`--coral-layers layer1 layer2 layer3`)
* Instantiates **3 distinct projection heads** ($176 \to 64$, $352 \to 64$, $2 \to 32$).

---

## 6. Mathematical Formulation

### 1. Sample Covariance in Projection Space
For a projected batch $\mathbf{P} \in \mathbb{R}^{B \times d_{\text{proj}}}$ with batch size $B$ and mean vector $\bar{\mathbf{p}} = \frac{1}{B} \sum_{i=1}^B \mathbf{p}_i$:
$$\mathbf{C}(\mathbf{P}) = \frac{1}{B - 1} (\mathbf{P} - \mathbf{1}\bar{\mathbf{p}}^T)^T (\mathbf{P} - \mathbf{1}\bar{\mathbf{p}}^T) \in \mathbb{R}^{d_{\text{proj}} \times d_{\text{proj}}}$$

### 2. Single-Head CORAL Loss
$$\mathcal{L}_{\text{CORAL}}^{(k)} = \frac{1}{4 \cdot d_{\text{proj}}^2} \|\mathbf{C}(\mathbf{P}_{\text{src}}^{(k)}) - \mathbf{C}(\mathbf{P}_{\text{tgt}}^{(k)})\|_F^2$$

### 3. Total Joint Optimization Objective
$$\min_{\Theta_{\text{HA02}}, \{\Phi_k\}_{k=1}^K} \mathcal{L}_{\text{MSE}}(Y_{\text{src}}, \hat{X}_{\text{src}}) + \alpha \cdot \mathcal{L}_{\text{SSIM}}(Y_{\text{src}}, \hat{X}_{\text{src}}) + \frac{\lambda_{\text{CORAL}}}{K} \sum_{k=1}^K \mathcal{L}_{\text{CORAL}}^{(k)}$$

where $\Theta_{\text{HA02}}$ are the parameters of the Transformer and Decoder, and $\Phi_k$ are the parameters of the $k$-th projection head.

---

## 7. Execution Commands & Artifacts

### Running Experiments
```bash
# Multi-layer Projection Head CORAL (Layer 1 + Layer 2 with 64-D projected subspace at SNR = 5 dB)
python run_CORALpHead_LS_Attention.py --snr 5 --coral-layers layer1 layer2 --domain-weight 0.5 --save-features

# Single-layer Projection Head on Transformer Encoder (Layer 1 only)
python run_CORALpHead_LS_Attention.py --snr 5 --coral-layers layer1 --domain-weight 0.5

# Source-only baseline (no domain adaptation)
python run_CORALpHead_LS_Attention.py --snr 5 --only-source
```

### Generated Artifacts in `results/`
* **`testChannel_source.mat` & `testChannel_target.mat`**: Held-out test channel arrays ready for MATLAB BER simulations.
* **`evaluation_results.mat`**: Summary scalar test metrics (NMSE dB, MMSE, SSIM).
* **`final_epoch.txt`**: Consolidated text evaluation report.
* **`training_history.mat`**: Complete loss and validation curves.
* **`extracted_features.mat`**: Both raw layer features (`features_{stage}_layer1_src`) and projected features (`features_{stage}_phead_layer1_src`) captured at begin, mid, and last training epochs.
* **`loss_total.pdf` & `val_nmse_db.pdf` & `target_test_sample_reconstruction.pdf`**: Visual training and channel reconstruction figures.