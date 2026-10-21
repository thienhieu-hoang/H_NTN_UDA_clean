# FDA (Fourier Domain Adaptation) End-to-End Workflow

This document details the complete end-to-end pipeline for channel estimation using **Fourier Domain Adaptation (FDA)** across three core stages:
1. **Pseudo Dataset Generation via FDA**
2. **Model Training on Pseudo Dataset**
3. **Inference & Evaluation on Real Target Dataset**

---

## 1. High-Level Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Pseudo Dataset Generation (FDA Translation)                                   │
│ Location: generatedChanPseudo/                                                         │
│                                                                                        │
│   Source Label (H_perfect)  ──┐                                                        │
│                               ├─► [FDA Fourier Amplitude Swap] ──► pseudoDataset (.mat)│
│   Target Inferred (LS+Attn) ──┘                                                        │
│                                                                                        │
│   • Preprocessing Options: Root Mean Square (RMS) | Min-Max | Standardize (std)        │
│   • Independence: FDA preprocessing is self-contained and decoupled from Stage 2/3.  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Model Training on Pseudo Dataset                                              │
│ Location: single_dataset/                                                              │
│                                                                                        │
│   pseudoDataset (.mat) ─────────► [Neural Network Training] ──► Trained ONNX Model     │
│                                   (LS+Attention / LI+DnCNN)                            │
│                                                                                        │
│   • Preprocessing Options: Root Mean Square (RMS) | Min-Max | Standardize (std)        │
│   • Golden Rule: MUST match Stage 3 preprocessing.                                     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Inference & Evaluation on Target Dataset                                      │
│ Location: inference/ (via cmd_auto_s1_2_3.ps1)                                         │
│                                                                                        │
│   Trained ONNX Model ───────────► [Model Inference] ──────────► Metrics & BER Plots    │
│   + Real Target Channel (H_prac)                                (inferredChannel.mat)  │
│                                                                                        │
│   • Preprocessing Options: Root Mean Square (RMS) | Min-Max | Standardize (std)        │
│   • Golden Rule: MUST match Stage 2 preprocessing.                                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Preprocessing Compatibility Rules

| Stage | Operation | Supported Preprocessing Options | Compatibility Rule |
| :--- | :--- | :--- | :--- |
| **Stage 1: FDA Generation** | FFT low-frequency spectrum swap $\rightarrow$ IFFT | • **RMS**<br>• **Min-Max**<br>• **Standardize** | **Independent**: Can differ from Stage 2 & 3. Chosen purely for best domain alignment. |
| **Stage 2: Model Training** | Train neural network on generated pseudo channel | • **RMS**<br>• **Min-Max**<br>• **Standardize** | **Strict Coupling**: **MUST be identical** to Stage 3 preprocessing. |
| **Stage 3: Target Inference**| Infer trained model on practical target channels | • **RMS**<br>• **Min-Max**<br>• **Standardize** | **Strict Coupling**: **MUST be identical** to Stage 2 preprocessing. |

> [!IMPORTANT]
> **Why Training & Inference Preprocessing Must Match:**  
> The neural network weights learn mathematical feature distributions, gradient scales, and activation boundaries based on the normalized data distribution during Stage 2. If inference in Stage 3 uses a different normalization scheme (e.g., training on Standardize but inferring with Min-Max or RMS), input activations will heavily saturate or collapse, degrading estimation NMSE and BER.
>
> **Why FDA Preprocessing Can Differ:**  
> FDA is an offline data translation step. Once the pseudo channels are reconstructed in spatial/frequency matrix form and saved to `.mat` files, the dataset is self-contained. The downstream training loader will normalize the dataset anew based on its own designated preprocessing rule.

---

## 3. Detailed Stage Breakdown

### Stage 1: Pseudo Dataset Generation (FDA Translation)
- **Script Location:** `generatedChanPseudo/` (e.g., `pseudoDatasetGen_Perfect_LSAttention.m`)
- **Inputs:**
  - **Source Ground Truth / Label**: $H_{\text{perfect}}$ from Source domain (e.g., TDL-A 100 ns).
  - **Target Practical / Inferred Channel**: Output of initial model on Target domain (e.g., DUR 300 ns inferred via LS+Attention).
- **Process:**
  1. Apply chosen FDA preprocessing to both channels (RMS, Min-Max, or Standardize).
  2. Compute 2D Fourier transform $\mathcal{F}(H)$ of both channels.
  3. Swap low-frequency amplitude spectrum within a central window (e.g., $13 \times 3$ or $13 \times 5$).
  4. Perform 2D inverse Fourier transform $\mathcal{F}^{-1}$ to obtain adapted channel matrices.
  5. Invert the preprocessing scaling back to original channel scale.
- **Output:** Stored under `generatedChanPseudo/<FDA_Dataset_Name>/` (or linked via junction).

---

### Stage 2: Training on Pseudo Dataset
- **Script Location:** `single_dataset/`
- **Inputs:** The pseudo dataset generated in Stage 1.
- **Process:**
  1. Load pseudo channel pairs $(H_{\text{in}}, H_{\text{target}})$.
  2. Normalize inputs and targets using the **designated training preprocessing** (`rms`, `minmax`, or `standardize`).
  3. Train estimator architecture (e.g., `LS+Attention` or `LI+DnCNN`).
  4. Export final and best model weights to ONNX format (`best_model.onnx`).
- **Signal File:** Generates `done_train.md` upon completion.

---

### Stage 3: Inference & Evaluation on Target
- **Script Location:** `inference/`
- **Execution Script:** `cmd_auto_s1_2_3.ps1` (or standalone `cmd_s1_inference.ps1`)
- **Inputs:**
  - Trained ONNX model from Stage 2.
  - Real Target dataset (e.g., `DUR300nsFix_...`).
- **Process:**
  1. Python ONNX runner applies the **same preprocessing** as Stage 2 to normalize input channels.
  2. Forward pass outputs channel predictions.
  3. De-normalize predictions back to physical complex channel units.
  4. Save `inferredChannel.mat` containing $H_{\text{infer}}$, NMSE, and SSIM.
  5. Run MATLAB Stage 2 & 3 synthesis scripts (`syn_metrics_withBER.m`, `syn_syn_compare_multiModels.m`) to generate BER curves and summary figures.
