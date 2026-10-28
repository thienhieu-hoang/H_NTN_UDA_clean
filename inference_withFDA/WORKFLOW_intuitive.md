# Intuitive Workflow Comparison: With Pseudo Dataset vs. Without Pseudo Dataset

This document provides a clear, physical, and intuitive explanation of the two Fourier Domain Adaptation (FDA) paradigms for wireless channel estimation.

---

## 1. Notation & Definitions

| Term | What it is | Real-World Example |
| :--- | :--- | :--- |
| **`source_label`** | Clean ground truth channel in Source domain ($H_{\text{perf}}^{\text{src}}$) | Clean TDL-A (100 ns delay spread, 70° elevation) |
| **`source_input`** | Noisy / practical channel estimate in Source domain ($H_{\text{LI}}^{\text{src}}$ or $H_{\text{LS}}^{\text{src}}$) | LS at pilots + Linear Interpolation (noisy) |
| **`model1`** | Neural network trained on `(source_input, source_label)` | Trained `LS+Attention` or `LI+DnCNN` |
| **`target_label`** | True channel in Target domain ($H_{\text{perf}}^{\text{tgt}}$) | Clean DUR (300 ns delay spread, 30° elevation) |
| **`target_input`** | Noisy channel estimate received in Target domain ($H_{\text{LI}}^{\text{tgt}}$ or $H_{\text{LS}}^{\text{tgt}}$) | What the receiver physically measures right now |

---

## 2. Approach 1: WITH Pseudo Dataset (Offline Retraining Paradigm)

*Goal: Teach a new network (`model2`) how the target environment behaves before deployment.*

```
Step 1: Get Initial Target Estimate
   target_input ────► [ model1 ] ────► target_outputA (Rough estimate of target channel)

Step 2: Generate Pseudo Ground-Truth (FDA)
   Base / Content / Phase (Clean Ground Truth) : source_label
   Macro-Style / Inner Window (Doppler & Delay) : target_outputA
   ─────────────────────────────────────────────────────────────
   pseudo_label = FDA( source_label <-- target_outputA )
   (Result: A 100% clean channel, but shaped with target Doppler shift & delay spread!)

Step 3: Simulate Transmission
   pseudo_label ────► [ Pilot Insertion + AWGN Noise ] ────► pseudo_input

Step 4: Train New Model
   Train model2 on (pseudo_input, pseudo_label)

Step 5: Direct Inference on Target
   target_input ────► [ model2 ] ────► Final Channel Estimate
   (No translation needed at test time: model2 already learned target statistics!)
```

> [!IMPORTANT]
> **Key Insight on Step 2:**  
> The base/phase comes from **`source_label`**, NOT `target_outputA`.  
> Why? Because `source_label` is perfectly clean (zero noise). We keep that clean quality, but replace its central frequencies with the Doppler/delay "style" of `target_outputA`.

---

## 3. Approach 2: WITHOUT Pseudo Dataset (Zero-Retraining / Pure Inference)

*Goal: Deploy the existing `model1` directly on the target dataset without training any new network.*

```
                 ┌────────────────────────────────────────────────────────┐
                 │                THE 4-STEP INFERENCE PIPELINE           │
                 └───────────────────────────┬────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ Step 1: Pre-FDA Input Translation (Protect Model Features)                               │
│                                                                                          │
│   target_input  ───► Keep Phase / Content (Current instantaneous channel)                │
│   source_input  ───► Take Inner Window (Macro Source Doppler & Delay Spread)             │
│   ──────────────────────────────────────────────────────────────────────                 │
│   pseudo_input2 = FDA( target_input  <--  source_input )                                 │
│                                                                                          │
│   • Why source_input (not source_label)?                                                 │
│     Because model1 was trained on noisy inputs (LS/LI), not clean labels.                 │
│     This aligns target input's delay/Doppler energy with what model1's layers expect.    │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ Step 2: Model Inference (Denoising & Interpolation)                                      │
│                                                                                          │
│   pseudo_input2 ───► [ model1 ] ───► target_outputB                                      │
│                                                                                          │
│   • What is target_outputB?                                                              │
│     A denoised and interpolated channel.                                                 │
│     However, because model1 applied its source priors, target_outputB has a              │
│     SOURCE DOMAIN BIAS (e.g., suppressed 300 ns multipath tails).                        │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ Step 3: Inverse Style Restoration / Post-FDA (Restore Target Physics)                    │
│                                                                                          │
│   target_outputB ──► Keep Denoised Details & Phase (Neural Network Output)               │
│   target_input   ──► Take Inner Window (True Target Doppler & Delay Spread)              │
│   ─────────────────────────────────────────────────────────────────────────              │
│   target_restored = FDA( target_outputB  <--  target_input )                             │
│                                                                                          │
│   • Why translate with target_input?                                                     │
│     target_input comes from the real target environment. Even though it is noisy at      │
│     individual subcarriers, its low-frequency Delay-Doppler spectrum has high cumulative │
│     SNR and contains the TRUE Doppler shift and multipath delay spread of the target!    │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ Step 4: Pilot Consistency Projection (Anchor to Measured Reality)                        │
│                                                                                          │
│   At known pilot locations (p_row, p_col):                                               │
│   Y_final[pilots] = (1 - alpha) * target_restored[pilots] + alpha * H_measured_LS        │
│                                                                                          │
│   • Why?                                                                                 │
│     At pilot subcarriers, we have physically measured values (Y_rx / X_tx).              │
│     Soft-blending (alpha ≈ 0.5) guarantees that neither neural network hallucination     │
│     nor FDA edge distortion shifts the channel away from true pilot observations.        │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## 4. The Central Window Paradox Explained

> **"In Step 1, we replace the target's center with the source, and in Step 3, we put the target's center back. Did we just do a redundant loop?"**

**No — we borrowed the Source's power profile just to let the model clean up the Phase, then returned the Target's true profile back:**

1. **Step 1 (The Disguise)**: We temporarily lend `source_input`'s central **amplitude** to the target channel. This prevents `model1` from going out-of-distribution, allowing it to successfully **denoise the complex phase ($\Phi$)** and reconstruct the outer 95% of high-frequency multipath.
2. **Step 3 (The Restoration)**: We take back `target_input`'s authentic central **amplitude** (real target Doppler & delay profile), but pair it with the **model's newly cleaned phase and full-grid denoising**.

$$\text{Final Channel} = \underbrace{A_{\text{target\_input}}}_{\text{Authentic Target Doppler/Delay}} \times \exp\Big(j \cdot \underbrace{\Phi_{\text{model1\_cleaned}}}_{\text{Denoised Complex Phase}}\Big)$$

---

## 5. Summary of Test-Time FDA Approaches (Without Pseudo Dataset Retraining)

### Mathematical Convention
$$\text{FDA}(A \leftarrow B) = \text{Keep Phase \& High Frequencies of } A, \quad \text{Replace Central Low Frequencies with } B$$

> [!NOTE]
> **Property of Nested Window Translation:**  
> If window size is identical, chaining FDA translations like:
> $$\text{FDA}\big(\text{target\_input} \leftarrow \text{FDA}(\text{pseudo\_input} \leftarrow \text{source\_label})\big) \equiv \text{FDA}(\text{target\_input} \leftarrow \text{source\_label})$$
> Because the inner mask completely overwrites the center pixels, any intermediate donor is ignored and only the final donor's center is retained.

### Quick Ranking & Recommendation Overview

| Approach | Name | Rating | Core Idea | Recommendation Level |
| :---: | :--- | :---: | :--- | :--- |
| **2.5** | **Iterative Self-Refinement** | ⭐⭐⭐⭐⭐ | 2-pass FDA; cleans noisy center with Pass 1 estimate | **Highest Recommendation** (Zero retraining) |
| **2.6** | **Hybrid Ensemble** | ⭐⭐⭐⭐⭐ | Combines `model2` (macro target) + `model1` (clean phase) | **Highest Recommendation** (When `model2` exists) |
| **2.1** | **Dual FDA (Source Input)** | ⭐⭐⭐⭐ | Pre-FDA disguise + Post-FDA restoration + PCP | **Strong Baseline** (Currently implemented) |
| **2.2** | **Dual FDA (Source Label)** | ⭐⭐⭐ | Uses 100% clean source center as disguise | **Good Alternative** (Clean center, but slight noise mismatch) |
| **2.4** | **Post-FDA Only** | ⭐⭐ | Raw forward pass, then forces target Doppler center | **Moderate** (Model may hallucinate on raw input) |
| **2.3** | **Straight Raw Inference** | ⭐ | No FDA adaptation at all | **Baseline Lower Bound** (Reference for paper) |

---

### Approach 2.1: Dual FDA with Source Input (Current Implemented Flow)
*Idea: Disguise target as practical source input, let model denoise phase, then restore target Doppler profile.*

1. **Pre-FDA Input**:
   $$\text{pseudo\_input2} = \text{FDA}(\text{target\_input} \leftarrow \text{source\_input})$$
   *(Donor: takes central Doppler/delay amplitude of `source_input`)*
2. **Model Forward**:
   $$\text{pseudo\_output2} = \text{model1}(\text{pseudo\_input2})$$
3. **Post-FDA Restoration**:
   $$\text{target\_restored} = \text{FDA}(\text{pseudo\_output2} \leftarrow \text{target\_input})$$
   *(Donor: takes back central Doppler/delay amplitude of `target_input`)*
4. **Pilot Anchoring (PCP)**:
   $$\text{final\_channel}[\text{pilots}] = (1 - \alpha) \cdot \text{target\_restored}[\text{pilots}] + \alpha \cdot H_{\text{LS}}^{\text{tgt}}$$

*Assessment*: ⭐⭐⭐⭐ **Very Good / Primary Baseline**. Input matches the noisy distribution `model1` was trained on. Highly stable.

---

### Approach 2.2: Dual FDA with Source Label (Clean Source Center)
*Idea: Use the 100% noiseless source ground truth as the disguise donor instead of noisy `source_input`.*

1. **Pre-FDA Input**:
   $$\text{pseudo\_input3} = \text{FDA}(\text{target\_input} \leftarrow \text{source\_label})$$
   *(Donor: takes central amplitude of clean `source_label`)*
2. **Model Forward**:
   $$\text{pseudo\_output3} = \text{model1}(\text{pseudo\_input3})$$
3. **Post-FDA Restoration**:
   $$\text{target\_restored} = \text{FDA}(\text{pseudo\_output3} \leftarrow \text{target\_input})$$
4. **Pilot Anchoring (PCP)**:
   $$\text{final\_channel} = \text{PCP}(\text{target\_restored}, H_{\text{LS}}^{\text{tgt}})$$

*Assessment*: ⭐⭐⭐ **Fair / Moderate**. Cleanest input center, but because `model1` was trained on noisy inputs, a noiseless center can slightly deviate from the model's expected noise statistics.

---

### Approach 2.3: Raw Baseline (Straight Inference)
*Idea: Pure cross-domain inference without any Fourier adaptation.*

1. **Model Forward**:
   $$\text{raw\_output} = \text{model1}(\text{target\_input})$$
2. **Pilot Anchoring (PCP)**:
   $$\text{final\_channel} = \text{PCP}(\text{raw\_output}, H_{\text{LS}}^{\text{tgt}})$$

*Assessment*: ⭐ **Lower Bound Baseline**. Required for scientific papers to prove how badly performance degrades when FDA is omitted.

---

### Approach 2.4: Post-FDA Only (Output Restoration)
*Idea: Do not disguise input; let `model1` predict raw, then inject target Doppler profile into the output.*

1. **Model Forward**:
   $$\text{raw\_output} = \text{model1}(\text{target\_input})$$
2. **Post-FDA Restoration**:
   $$\text{target\_restored} = \text{FDA}(\text{raw\_output} \leftarrow \text{target\_input})$$
3. **Pilot Anchoring (PCP)**:
   $$\text{final\_channel} = \text{PCP}(\text{target\_restored}, H_{\text{LS}}^{\text{tgt}})$$

*Assessment*: ⭐⭐ **Moderate**. Faster (only 1 FDA operation), but because `model1` receives out-of-distribution raw target inputs, its high-frequency phase reconstruction is suboptimal.

---

### Approach 2.5: Iterative Self-Refinement (2-Pass FDA)
*Idea: Use the denoised output from Approach 2.1 as a much cleaner style donor for a second pass.*

1. **Pass 1**: Run **Approach 2.1** to get initial denoised channel $\hat{H}^{(1)}$.
2. **Pre-FDA Input (Pass 2)**:
   $$\text{pseudo\_input}^{(2)} = \text{FDA}(\text{target\_input} \leftarrow \hat{H}^{(1)})$$
   *(Donor: takes central amplitude from denoised $\hat{H}^{(1)}$ rather than noisy raw inputs)*
3. **Model Forward (Pass 2)**:
   $$\text{output}^{(2)} = \text{model1}(\text{pseudo\_input}^{(2)})$$
4. **Final Post-FDA & PCP**:
   $$\text{final\_channel} = \text{PCP}\Big(\text{FDA}(\text{output}^{(2)} \leftarrow \hat{H}^{(1)}), \ H_{\text{LS}}^{\text{tgt}}\Big)$$

*Assessment*: ⭐⭐⭐⭐⭐ **Top Recommendation (Zero Retraining)**. Eliminates measurement noise from the central Doppler bins while retaining full target geometry. Gives the lowest NMSE among pure test-time methods.

---

### Approach 2.6: Hybrid with Pseudo Model (Ensemble `model2` + `model1`)
*Idea: When you also have `model2` (trained on pseudo-dataset), combine both models at test time.*

1. **Initial Target Guess**:
   $$\hat{H}_{\text{m2}} = \text{model2}(\text{target\_input})$$
2. **Fine Phase & Multipath Denoising**:
   $$\text{pseudo\_in} = \text{FDA}(\text{target\_input} \leftarrow \hat{H}_{\text{m2}})$$
   $$\hat{H}_{\text{m1}} = \text{model1}(\text{pseudo\_in})$$
3. **Final Blend**:
   $$\text{final\_channel} = \text{PCP}\Big(\text{FDA}(\hat{H}_{\text{m1}} \leftarrow \hat{H}_{\text{m2}}), \ H_{\text{LS}}^{\text{tgt}}\Big)$$

*Assessment*: ⭐⭐⭐⭐⭐ **Top Recommendation (Hybrid Ensemble)**. Leverages `model2`'s target domain familiarity while using `model1`'s clean source-trained weights for superior phase denoising.

---

## 6. Recommended Test-Time FDA Approaches for `model2` (Trained on Pseudo Dataset)

`model2` was retrained offline on:
$$\text{pseudo\_label} = \text{FDA}(\text{source\_label} \leftarrow \text{target\_outputA})$$

Unlike `model1`, **`model2` already learned the target domain's macro Doppler shift and delay spread**. However, at test time, `model2` can still suffer from:
1. **Residual Pseudo Bias**: Its training labels came from `target_outputA` (which had imperfect initial estimation error).
2. **Phase / Noise Mismatch**: The phase in `pseudo_label` was originally derived from `source_label`.
3. **Pilot Hallucination**: Neural networks can hallucinate even when adapted.

Here are the recommended test-time FDA approaches when deploying `model2`:

---

### Approach 3.1: Straight Inference + Pilot Consistency (The Standard Baseline)
*Idea: Let `model2` predict directly (no Pre-FDA disguise needed), then anchor pilots with real measurements.*

1. **Model Forward**:
   $$\hat{H}_{\text{m2}} = \text{model2}(\text{target\_input})$$
2. **Pilot Anchoring (PCP)**:
   $$\text{final\_channel}[\text{pilots}] = (1 - \alpha) \cdot \hat{H}_{\text{m2}}[\text{pilots}] + \alpha \cdot H_{\text{LS}}^{\text{tgt}}$$

*Assessment*: ⭐⭐⭐ **Standard baseline**. Simple and fast, but leaves any residual estimation error from the offline pseudo-generator uncorrected.

---

### Approach 3.2: Post-FDA Bias Correction (Highly Recommended)
*Idea: Since `model2` was trained on imperfect `target_outputA` centers, replace its output center with the physical `target_input`.*

1. **Model Forward**:
   $$\hat{H}_{\text{m2}} = \text{model2}(\text{target\_input})$$
   *(Model reconstructs full grid and denoises target phase)*
2. **Post-FDA Restoration**:
   $$\text{target\_corrected} = \text{FDA}(\hat{H}_{\text{m2}} \leftarrow \text{target\_input})$$
   *(Takes back the real, physical low-frequency Doppler center directly measured by the receiver)*
3. **Pilot Anchoring (PCP)**:
   $$\text{final\_channel} = \text{PCP}(\text{target\_corrected}, H_{\text{LS}}^{\text{tgt}})$$

*Assessment*: ⭐⭐⭐⭐⭐ **Top Recommendation for `model2`**. Fixes any residual bias inherited from `target_outputA` during pseudo dataset generation.

---

### Approach 3.3: Pre-FDA Alignment to Training Pseudo Distribution
*Idea: Make `target_input` look exactly like the `pseudo_input` that `model2` saw during training.*

1. **Pre-FDA Input**:
   $$\text{aligned\_input} = \text{FDA}(\text{target\_input} \leftarrow \text{pseudo\_prototype})$$
   *(Donor: prototype Delay-Doppler spectrum of `pseudo_input` used in training)*
2. **Model Forward**:
   $$\hat{H}_{\text{pred}} = \text{model2}(\text{aligned\_input})$$
3. **Post-FDA & PCP**:
   $$\text{final\_channel} = \text{PCP}\Big(\text{FDA}(\hat{H}_{\text{pred}} \leftarrow \text{target\_input}), \ H_{\text{LS}}^{\text{tgt}}\Big)$$

*Assessment*: ⭐⭐⭐⭐ **Very Good**. Closes any remaining domain gap between the simulated `pseudo_input` noise distribution and real `target_input`.

---

### Approach 3.4: Teacher-Student Cross-Refinement (`model2` Macro Prior + `model1` Phase Denoising)
*Idea: `model2` is the expert on target macro geometry; `model1` was trained on clean source ground truth and has sharper multipath denoising.*

1. **Step 1 (Macro Target Guess via `model2`)**:
   $$\hat{H}_{\text{m2}} = \text{model2}(\text{target\_input})$$
2. **Step 2 (Disguise for `model1` using `model2`'s Clean Estimate)**:
   $$\text{pseudo\_in} = \text{FDA}(\text{target\_input} \leftarrow \hat{H}_{\text{m2}})$$
   $$\hat{H}_{\text{refined}} = \text{model1}(\text{pseudo\_in})$$
3. **Step 3 (Post-FDA & Anchor)**:
   $$\text{final\_channel} = \text{PCP}\Big(\text{FDA}(\hat{H}_{\text{refined}} \leftarrow \text{target\_input}), \ H_{\text{LS}}^{\text{tgt}}\Big)$$

*Assessment*: ⭐⭐⭐⭐⭐ **Highest Potential Ensemble**. Combines `model2`'s domain familiarity with `model1`'s clean multipath reconstruction power.

---

## 7. Pilot Anchoring (Pilot Consistency Projection - PCP): Deep Dive

### 7.1 What is Pilot Anchoring?
In OFDM systems, the receiver does not need to guess what was transmitted at pilot subcarriers. Known reference symbols $X_p$ are sent at specific time-frequency coordinates $\mathcal{P} = \{(p_{\text{row}}, p_{\text{col}})\}$.

The physical Least Squares (LS) measurement at these pilot points is:
$$H_{\text{LS}}[p] = \frac{Y[p]}{X[p]} = H_{\text{true}}[p] + \frac{Z[p]}{X[p]}$$

**Pilot Consistency Projection (PCP)** is a post-processing operation that forces or blends the adapted neural network estimate $\hat{H}$ at pilot coordinates back toward the physical measurement $H_{\text{LS}}$, while leaving data subcarriers untouched.

---

### 7.2 How Does It Work Mathematically?

Given the reconstructed full grid $\hat{H}_{\text{restored}} \in \mathbb{C}^{132 \times 14}$ after Post-FDA:

$$\hat{H}_{\text{final}}[k, l] = \begin{cases}
(1 - \alpha) \cdot \hat{H}_{\text{restored}}[k, l] + \alpha \cdot H_{\text{LS}}^{\text{tgt}}[k, l], & \text{if } (k, l) \in \mathcal{P} \text{ (Pilot subcarriers)} \\[6pt]
\hat{H}_{\text{restored}}[k, l], & \text{if } (k, l) \notin \mathcal{P} \text{ (Data subcarriers)}
\end{cases}$$

Where $\alpha \in [0, 1]$ is the **Pilot Consistency Factor**:
- **$\alpha = 1.0$ (Hard Projection)**: Overwrites the model's pilot predictions with $100\%$ physical measurements. *(Best at high SNR $\ge 10\text{ dB}$ where $H_{\text{LS}}$ has very low noise)*.
- **$\alpha = 0.5$ (Soft Blending - Recommended Default)**: Averages the neural model's denoised prediction with the physical observation. Combines model noise suppression with physical observation grounding.
- **$\alpha = 0.0$ (Disabled)**: Accepts purely the model's predicted values without pilot correction.

---

### 7.3 Why Do We Need Pilot Anchoring Specifically AFTER Post-FDA?

There are **three fundamental physical reasons** why PCP is critical after Post-FDA:

#### Reason 1: 2D-FFT/IFFT Global Energy Leakage Perturbs Known Pilots
- FDA is performed in the **Delay-Doppler Fourier domain**:
  $$\text{Time-Frequency Grid } (132 \times 14) \xrightarrow{\text{2D-FFT}} \text{Delay-Doppler} \xrightarrow{\text{Amplitude Swap}} \xrightarrow{\text{2D-IFFT}} \text{Time-Frequency Grid}$$
- The 2D-IFFT is a **global linear operation**: changing the central Delay-Doppler amplitudes redistributes energy across **every single subcarrier and OFDM symbol** in the time-frequency grid.
- Consequently, even if the neural model had predicted perfect pilot values, the Post-FDA Fourier synthesis slightly perturbs the values at pilot locations. **PCP directly undoes this Fourier leakage at known pilot positions.**

#### Reason 2: Ground-Truth Anchor Against Model Hallucination
- Under cross-domain inference (e.g., $100\text{ ns} \to 300\text{ ns}$ delay spread, $70^\circ \to 30^\circ$ elevation), neural networks can experience residual out-of-distribution drift.
- While data subcarriers have no ground truth and must rely on the network's interpolation, **pilots are physically measured facts ($Y_p / X_p$)**.
- Anchoring to pilots provides an absolute physical "grounding" that prevents neural network hallucination from propagating into the equalized constellation symbols.

#### Reason 3: Noise Filtering vs. Bias Elimination ($\alpha = 0.5$ Balance)
- At low SNRs (e.g., $-10\text{ dB}$ to $0\text{ dB}$), raw $H_{\text{LS}}$ contains significant Gaussian noise $\sigma_n^2$.
- The neural network acts as a multi-layer Wiener/denoising filter, but may suffer from small amplitude bias.
- Setting $\alpha = 0.5$ achieves the optimal Bayesian compromise:
  $$\hat{H}_{\text{final}}[\text{pilots}] = \frac{1}{2}\underbrace{\hat{H}_{\text{restored}}}_{\text{Low Variance (Denoised)}} + \frac{1}{2}\underbrace{H_{\text{LS}}}_{\text{Zero Bias (Physically True)}}$$
  This simultaneously lowers estimation variance and eliminates domain bias.
