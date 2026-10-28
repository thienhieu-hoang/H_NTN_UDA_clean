# Test-Time FDA & Cross-Domain Inference: Workflows, Post-Processing & Recommendations

## 1. Executive Summary & Quick Recommendation Matrix

When performing cross-domain channel estimation using **Fourier Domain Adaptation (FDA)**, there are two fundamental paradigms:
1. **Training-Time Adaptation (Pseudo Dataset Retraining)**: Synthesizes pseudo labels offline, trains a new network, and infers on target data.
2. **Test-Time Adaptation (Zero Retraining / Pure Inference)**: Employs a network already trained on the Source domain, and uses FDA at inference time on the target inputs/outputs.

Below is the comparative breakdown of the candidate workflows:

| Workflow Paradigm | Model Used | Input Pre-Processing | Output Post-Processing | Retraining Needed? | Expected Performance | Complexity |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Workflow A (Baseline)** | Source-trained ($f_{\theta_{\text{src}}}$) | Raw Target input ($X_{\text{tgt}}$) | None | No | Low (Domain Gap degrades BER/NMSE) | Minimal |
| **Workflow B (Existing Pseudo)** | Pseudo-trained ($f_{\theta_{\text{pseudo}}}$) | Raw Target input ($X_{\text{tgt}}$) | None | Yes | **High** (Network learned target profile) | High (Data gen + train) |
| **Workflow C (Pre-FDA Only)** | Source-trained ($f_{\theta_{\text{src}}}$) | $X_{\text{tgt} \to \text{src}}$ (FDA swapped) | None | **No** | Moderate (Input aligned, output has source bias) | Low |
| **Workflow D (Dual FDA + Pilot Consistency)** ⭐ *(Top Zero-Retrain)* | Source-trained ($f_{\theta_{\text{src}}}$) | $X_{\text{tgt} \to \text{src}}$ (Pre-FDA) | **Post-FDA** $\hat{Y} \leftarrow X_{\text{tgt}}$ + **Pilot Consistency** | **No** | **Very High** (Zero-retrain benchmark winner) | Moderate |
| **Workflow E (Post-FDA Only)** | Source-trained ($f_{\theta_{\text{src}}}$) | Raw Target input ($X_{\text{tgt}}$) | Post-FDA style swap | **No** | Moderate-High (Simple and effective) | Low |

---

## 2. In-Depth Analysis of Your Specific Questions

### Question 1: Trained Model on Source Domain + FDA Translated Input
> *"use the trained model on source domain; input is the target input translated to source input/label (not sure the output is really the target label)"*

#### The Physical & Mathematical Reality:
1. **The Model's Priors**: The source model $f_{\theta_{\text{src}}}$ was trained to approximate the conditional distribution $P(Y_{\text{src}} \mid X_{\text{src}})$. It learned:
   - The multipath delay profile and maximum delay spread $\tau_{\max}$ of the Source (e.g., TDL-A 100 ns).
   - The Doppler frequency profile $\nu_D$ of the Source (e.g., 70° elevation, velocity $v_{\text{src}}$).
   - The pilot interpolation pattern corresponding to the Source correlation matrix.

2. **What Pre-FDA Translation Does ($X_{\text{tgt} \to \text{src}}$)**:
   By replacing the low-frequency Fourier amplitude of $X_{\text{tgt}}$ with a Source reference spectrum $\bar{A}_{\text{src}}$:
   $$\mathcal{F}(X_{\text{tgt} \to \text{src}}) = \left( (1 - M) \odot A(X_{\text{tgt}}) + M \odot \bar{A}_{\text{src}} \right) e^{j \Phi(X_{\text{tgt}})}$$
   - **Advantage**: The model receives an input whose macro-spectral energy matches its training distribution. Attention maps (in Transformers) and convolutional filters (in DnCNN) will **not** suffer catastrophic out-of-distribution saturation.
   - **The Dilemma (Your Concern)**: **The output is NOT purely the target channel**. Because $f_{\theta_{\text{src}}}$ has strong internal priors, its output $\hat{Y}_{\text{pred}}$ will act as a *source-like* denoiser. If the target has a larger delay spread (e.g., DUR 300 ns vs. A 100 ns), $f_{\theta_{\text{src}}}$ will tend to suppress the long-delay multipath tails, treating them as noise!

**Conclusion for Q1**: Pre-FDA translation protects the model's feature representations, but without post-processing, the estimated channel retains a *source domain spectral bias*.

---

### Question 2: Trained Model on Pseudo Dataset
> *"use the trained model on pseudo dataset; input is target input vs. input translated to source"*

1. **Option 2A: Feed Raw Target Input ($X_{\text{tgt}}$)**:
   - **Why this works**: In the pseudo dataset pipeline (`generatedChanPseudo/`), the pseudo ground truth $H_{\text{pseudo}}$ was constructed by injecting target Doppler/delay amplitude into source channels ($H_{\text{perf}}^{\text{src}} \leftarrow H_{\text{infer}}^{\text{tgt}}$).
   - The network $f_{\theta_{\text{pseudo}}}$ was explicitly trained on inputs derived from $H_{\text{pseudo}}$, which **already reflect target domain characteristics**.
   - Therefore, at inference time, feeding raw $X_{\text{tgt}}$ is optimal because $X_{\text{tgt}}$ shares the exact low-frequency spectral structure that $f_{\theta_{\text{pseudo}}}$ was trained to reconstruct.

2. **Option 2B: Translate $X_{\text{tgt}} \to X_{\text{src}}$ and feed to Pseudo Model**:
   - **Harmful / Counterproductive**: Doing this transforms the input *away* from the target style back to the source style, which directly conflicts with what the pseudo model learned.

**Conclusion for Q2**: For a pseudo-trained model, **always feed the raw target input $X_{\text{tgt}}$**. Pre-translating to source input will degrade accuracy.

---

### Question 3: Post-Processing & Inverse FDA
> *"How about using post-processing? After get the output, can we do inverse FDA? Inverse FDA the target output from the source input/label? FDA translate target output to target input?"*

**YES! Post-processing is mathematically sound and physically justified in channel estimation.**

#### Why Post-Processing Works in Wireless Channels:
A neural network channel estimator performs two primary functions:
1. **Denoising** (removing AWGN noise).
2. **Interpolation** (reconstructing data subcarriers/symbols from sparse pilots).

Under domain shift, the neural network's **high-frequency output** is valuable (it denoised the fine multipath structure), but its **low-frequency profile** may be biased toward the source domain. Meanwhile, the raw target input $X_{\text{tgt}}$:
- Is noisy at individual subcarriers, BUT
- Its low-frequency Delay-Doppler spectrum has **high cumulative SNR** and contains the **true Doppler shift and delay spread of the target environment**.

#### Recommended Post-Processing Techniques:

```
                  ┌────────────────────────────────────────────────────────┐
                  │            POST-PROCESSING PIPELINE OPTIONS            │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
             ┌───────────────────────────────┴──────────────────────────────┐
             ▼                                                              ▼
┌───────────────────────────────┐                             ┌───────────────────────────────┐
│ Post-Processing Option 1:     │                             │ Post-Processing Option 2:     │
│ Inverse FDA (Style Restore)   │                             │ Hard Pilot Consistency        │
│                               │                             │ Projection (PCP)              │
│ Swap low frequencies of       │                             │ Replace predicted pilots with │
│ model output with target      │                             │ measured LS pilots:           │
│ input amplitude:              │                             │                               │
│ Y_final = FDA(Y_hat <- X_tgt) │                             │ Y_final(pilots) = Y_meas(pilots)
└───────────────────────────────┘                             └───────────────────────────────┘
```

#### Technique 1: Inverse FDA (Target Style Restoration)
Take the model's raw output $\hat{Y}$ and re-inject the target's low-frequency amplitude spectrum:
$$\mathcal{F}(\hat{Y}_{\text{final}}) = \left( (1 - M) \odot A(\hat{Y}) + M \odot A(X_{\text{tgt}}) \right) e^{j \Phi(\hat{Y})}$$
$$\hat{Y}_{\text{final}} = \mathcal{F}^{-1}(\mathcal{F}(\hat{Y}_{\text{final}}))$$
- **Effect**: Retains the network's denoised high-frequency components while guaranteeing that the central delay-Doppler energy distribution perfectly matches the target channel.

#### Technique 2: Hard Pilot Consistency Projection (PCP)
In OFDM NTN systems, at pilot locations $(r_p, c_p)$, the receiver observed the received pilot symbol $Y_p$:
$$H_{\text{LS}}(r_p, c_p) = \frac{Y_p}{X_p}$$
Even after neural network estimation and FDA:
$$\hat{Y}_{\text{final}}(r_p, c_p) = (1 - \alpha) \cdot \hat{Y}_{\text{final}}(r_p, c_p) + \alpha \cdot H_{\text{LS}}(r_p, c_p)$$
where $\alpha \in [0, 1]$ depends on the SNR (at high SNR $>15\text{ dB}$, $\alpha \approx 0.8$; at low SNR $<0\text{ dB}$, $\alpha \approx 0.2$).

---

## 3. Recommended Workflow Architecture

If your objective is **Test-Time Adaptation (No pseudo dataset generation, no network retraining)**, the recommended pipeline is:

### The "Dual-Stage FDA + Pilot Consistency" Pipeline (Workflow D)

```
[Raw Target Test Channel]
       │
       ▼
[Step 1: Pre-FDA Translation]
   X_tgt_trans = FDA( X_tgt  <--  A_src_prototype )
   • Replaces low frequencies with precomputed Source mean amplitude spectrum.
   • Prevents Transformer attention saturation and CNN filter divergence.
       │
       ▼
[Step 2: Source Model Inference]
   Y_hat = Model_Source( X_tgt_trans )
   • Evaluates on trained ONNX model (LI_DnCNN or LS_Attention).
   • Neural network performs robust denoising and pilot interpolation.
       │
       ▼
[Step 3: Post-FDA Inverse Style Restoration]
   Y_restored = FDA( Y_hat  <--  X_tgt )
   • Re-injects authentic target Doppler/delay amplitude into model output.
       │
       ▼
[Step 4: Pilot Consistency Projection]
   Y_final(pilot_idx) = (1 - alpha) * Y_restored(pilot_idx) + alpha * H_LS(pilot_idx)
       │
       ▼
[Step 5: MATLAB Evaluation]
   syn_metrics_withBER.m (NMSE, SSIM, EVM, BER)
```

---

## 4. Other Advanced Test-Time Alternatives Worth Exploring

If you want to push test-time adaptation further without retraining, consider these two state-of-the-art approaches:

### Alternative 1: Test-Time Adaptation via Pilot Loss (TTA-Pilot)
- **Concept**: Use the 88 pilot symbols as an unsupervised self-supervision signal at inference time.
- **Mechanism**:
  1. For a test batch of target channels, compute the loss on the known pilot locations:
     $$\mathcal{L}_{\text{pilot}} = \frac{1}{N_p} \sum_{k=1}^{N_p} \|\hat{Y}(r_k, c_k) - H_{\text{LS}}(r_k, c_k)\|^2$$
  2. Perform $1 \sim 3$ gradient descent steps to update only the LayerNorm / BatchNorm parameters or a lightweight adapter head.
  3. Perform final inference.
- **Pros**: Adapts network weights to the new channel statistics on the fly.
- **Cons**: Slower inference speed due to backprop at test time.

### Alternative 2: Bandpass / Multi-Window FDA
- Instead of a fixed rectangular $13 \times 3$ or $13 \times 5$ window, use a smooth Hann/Gaussian 2D window (`fda_create_smooth_window.m`).
- Prevents spectral leakage (ringing artifacts) along the Delay-Doppler boundary during IFFT.

---

## 5. Proposed File Organization in `inference_withFDA/`

To keep everything clean, reproducible, and aligned with repository standards:

```
inference_withFDA/
├── FDA_INFERENCE_WORKFLOW_GUIDE.md   # This specification document
├── fda_test_time_utils.py           # Pre-FDA, Post-FDA, and PCP functions
├── precompute_source_spectrum.py    # Precomputes source prototype spectrum (mean A_src)
├── inference_onnx_grid_fda.py       # Stage 1: ONNX inference for LI_DnCNN with FDA modes
├── inference_onnx_ls_fda.py         # Stage 1: ONNX inference for LS_Attention with FDA modes
├── cmd_auto_s1_2_3.ps1              # Unified automation (Stage 1 -> Stage 2 -> Stage 3)
├── syn_metrics_withBER.m            # Stage 2: MATLAB BER and NMSE computation
└── syn_syn_compare_multiModels.m    # Stage 3: Comparison curves (BER vs SNR, NMSE vs SNR)
```

### CLI Flag Design for `cmd_auto_s1_2_3.ps1`:
```powershell
# FDA Mode selection:
#   "none"     : Pure baseline inference
#   "pre"      : Input FDA translation only
#   "post"     : Output Inverse FDA restoration only
#   "dual"     : Pre-FDA translation + Post-FDA restoration (Recommended)
#   "dual_pcp" : Dual FDA + Pilot Consistency Projection
$fdaMode = "dual_pcp"
$fdaWindow = @(13, 3)
```
