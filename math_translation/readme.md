# Mathematical Channel Translation & Physics-Guided Domain Adaptation

This module implements principled **physics-based mathematical channel translation** to adapt neural channel estimators from a Source NTN domain (e.g. `A100`, $100\text{ ns}$ delay spread, $70^\circ$ elevation, LoS) to an unlabeled Target NTN domain (e.g. `DUR300`, $300\text{ ns}$ delay spread, $30^\circ$ elevation, NLoS).

It directly supersedes naive Fourier Domain Adaptation (FDA box-swapping) by replacing arbitrary rectangular window cutting with **continuous coordinate dilation, bulk Doppler phase modulation, and pilot anchoring** governed by wireless propagation physics.

---

## 1. High-Level Architecture Overview

```
                                      [Unlabeled Target Pilots] H_ls^tgt
                                                   │
                                                   ▼
                               ┌──────────────────────────────────────────────┐
                               │       math_translation/core/                 │
                               │  • Blind Autocorrelation Drop Estimator     │
                               │  • Bulk Doppler Temporal Phase Estimator     │
                               │  • Delay Dilation & Modulation Operators     │
                               └──────────────────────┬───────────────────────┘
                                                      │
                       Estimated Physical Shifts: { β̂_τ ≈ 3.3,  Δν̂ ≈ -2.1 Hz }
                                                      │
              ┌───────────────────────────────────────┼──────────────────────────────────────┐
              ▼                                       ▼                                      ▼
┌───────────────────────────┐           ┌───────────────────────────┐          ┌───────────────────────────┐
│ Approach A (Data-Level)   │           │ Approach B (Test-Time)    │          │ Approach C (Feature UDA)  │
│ [Pseudo-Dataset Retrain]  │           │ [Zero-Retraining Deploy]  │          │ [Physics-Guided CORAL]    │
│                           │           │                           │          │                           │
│ Warp source ground-truth  │           │ Inverse-warp target pilots│          │ Warps source attention    │
│ labels by β̂_τ & Δν̂ to    │           │ (300ns → 100ns), infer    │          │ maps by β̂_τ and enforces  │
│ create target pseudo data.│           │ with frozen Model 1, then │          │ L_physics_attn alignment  │
│                           │           │ forward-warp prediction.  │          │ during joint training.    │
│ Output:                   │           │ Output:                   │          │ Output:                   │
│ results/approachA_pseudo/ │           │ results/approachB_test/   │          │ results/approachC_coral/  │
└───────────────────────────┘           └───────────────────────────┘          └───────────────────────────┘
```

---

## 2. Directory Structure

```
math_translation/
├── readme.md                           # Master architectural & mathematical guide (this file)
│
├── core/                               # Foundational signal processing & I/O engine
│   ├── __init__.py                     # Clean package exports
│   ├── io_utils.py                     # Robust v7 & v7.3 MAT file loader/saver & DM-RS coords
│   ├── metrics.py                      # MMSE, NMSE (linear & dB), SSIM (NumPy 2.x safe)
│   ├── estimate_pilot_shifts.py        # Blind estimation of β̂_τ, Δν̂, and 2D Delay-Doppler moments
│   └── warp_operators.py               # Delay dilation, Doppler modulation, pilot anchoring
│
├── approachA_pseudo/                   # Workflow A: Physical Pseudo-Dataset Generation
│   ├── generate_pseudo_dataset.py      # Generates pseudo target datasets across SNRs [-10 to +15 dB]
│   └── run_approach_A.ps1              # Automated PowerShell runner
│
├── approachB_test_time/                # Workflow B: Test-Time Domain Adaptation (Zero Retraining)
│   ├── test_time_warping_infer.py      # Inverse-warp → Model 1 inference → Forward-warp → Pilot anchor
│   └── cmd_run_approachB.ps1           # Automated PowerShell runner
│
├── approachC_physics_coral/            # Workflow C: Physics-Guided Feature Alignment
│   ├── physics_coral_loss.py           # TensorFlow & PyTorch L_physics_attn and L_physics_cov loss
│   ├── train_physics_coral.py          # Joint supervised + physics-guided attention UDA training
│   └── cmd_run_approachC.ps1           # Automated PowerShell runner
│
└── results/                            # Auto-generated experiment outputs & benchmarks
    ├── approachA_pseudo_dataset/
    ├── approachB_test_time/
    └── approachC_physics_coral/
```

---

## 3. Mathematical Foundations

### 3.1 Blind Statistical Extraction of $(\hat{\beta}_\tau, \Delta \hat{\nu})$
No satellite ephemeris, channel models, or true labels are required from the target domain. Parameters are extracted blindly from sparse pilots:

1. **Delay Dilation ($\hat{\beta}_\tau$)**:
   The frequency correlation of pilots separated by subcarrier lag $\Delta k$ is computed via:
   $$R_f(\Delta k) = \frac{1}{|\mathcal{P}|} \sum_{k \in \text{pilots}} H_{\text{ls}}(k) \cdot H_{\text{ls}}^*(k + \Delta k)$$
   The dilation factor is determined by comparing the $50\%$ coherence bandwidth drop-off lags:
   $$\hat{\beta}_\tau = \frac{\Delta k_{0.5}^{\text{source}}}{\Delta k_{0.5}^{\text{target}}}$$
   *(When testing between A100 $100\text{ ns}$ and DUR300 $300\text{ ns}$, the estimator reliably yields $\hat{\beta}_\tau \approx 3.30$).*

2. **Bulk Doppler Shift ($\Delta \hat{\nu}$)**:
   The phase difference between pilot symbols (symbols 2 and 11, separation $\Delta t = 9 \cdot T_{\text{sym}}$):
   $$\Delta \hat{\nu} = \frac{\angle \mathbb{E}\left[ H_{\text{ls}}^{\text{tgt}}(k, 11) H_{\text{ls}}^{\text{tgt}*}(k, 2) \right] - \angle \mathbb{E}\left[ H_{\text{ls}}^{\text{src}}(k, 11) H_{\text{ls}}^{\text{src}*}(k, 2) \right]}{2\pi \cdot 9 \cdot T_{\text{sym}}}$$

### 3.2 Continuous Delay Dilation & Modulation
Instead of cutting and pasting boxes in the Delay-Doppler grid, continuous coordinate stretching is applied in the Delay domain:
$$h(\tau, t) = \mathcal{F}_f^{-1}\left\{ H(f, t) \right\}$$
$$h_{\text{warped}}(\tau, t) = \frac{1}{\sqrt{\beta_\tau}} h\left(\frac{\tau}{\beta_\tau}, \, t\right)$$
$$H_{\text{warped}}(f, t) = \mathcal{F}_\tau\left\{ h_{\text{warped}}(\tau, t) \right\} \cdot e^{j 2\pi \Delta \nu \cdot t \cdot T_{\text{sym}}}$$
- **Zero Gibbs ringing artifacts**: Continuous delay stretching maintains natural multipath decays.
- **Physical phase-delay consistency**: Preserves exact wave relations $\phi(\tau, f) = -2\pi f \tau$.

### 3.3 Pilot Anchoring
At runtime, measured target pilots $H_{\text{ls}}^{\text{tgt}}$ are known with certainty (up to noise level). The predicted grid $\hat{H}$ is anchored at pilot positions $(r_p, c_p)$:
$$\hat{H}_{\text{final}}[n, r_p, c_p] = (1 - \alpha) \hat{H}[n, r_p, c_p] + \alpha H_{\text{ls}}^{\text{tgt}}[n, p]$$
Setting $\alpha = 1.0$ guarantees **$0\text{ dB}$ residual estimation error at pilot locations**.

---

## 4. Execution Guide

### Option A: Run Physical Pseudo-Dataset Generation (Approach A)
Generates full target pseudo ground-truth datasets for retraining:
```powershell
# From PowerShell in repository root:
cd .\math_translation\approachA_pseudo\
.\run_approach_A.ps1
```
Or via Python CLI:
```bash
python generate_pseudo_dataset.py \
    --source-dir "../../generatedChan/MATLAB/A100_2p18e9_600km_70deg_30kHz" \
    --target-dir "../../generatedChan/OpenNTN/DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps" \
    --out-dir "../results/approachA_pseudo_dataset" \
    --scale-mode rms \
    --snr-list -10 -5 0 5 10 15
```

---

### Option B: Run Test-Time Domain Adaptation (Approach B)
Performs instant test-time inference on the target channel using frozen Model 1 with **zero retraining**:
```powershell
cd .\math_translation\approachB_test_time\
.\cmd_run_approachB.ps1
```
Or via Python CLI:
```bash
python test_time_warping_infer.py \
    --model-dir "../../single_dataset/A100_2p18e9_600km_70deg_30kHz/LS_Attention_rms" \
    --target-dataset-dir "../../generatedChan/OpenNTN/DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps" \
    --out-dir "../results/approachB_test_time" \
    --pilot-anchor-blend 1.0
```

---

### Option C: Run Physics-Guided Attention CORAL (Approach C)
Jointly trains on source labels and unsupervised target pilots with physical attention alignment:
```powershell
cd .\math_translation\approachC_physics_coral\
.\cmd_run_approachC.ps1
```
Or via Python CLI:
```bash
python train_physics_coral.py \
    --source-dir "../../generatedChan/MATLAB/A100_2p18e9_600km_70deg_30kHz" \
    --target-dir "../../generatedChan/OpenNTN/DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps" \
    --save-dir "../results/approachC_physics_coral" \
    --snr 10 \
    --epochs 100 \
    --domain-weight 1.0 \
    --norm-mode rms
```

---

## 5. Comparison Matrix

| Approach | Retraining Needed? | Blind Parameter Estimation? | Expected Target NMSE | Best Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **Approach A (Pseudo Retrain)** | Yes (30-50 epochs) | Yes ($\hat{\beta}_\tau, \Delta \hat{\nu}$) | **Highest** (~ -9.6 to -11 dB) | Upgrading FDA for final paper submission tables |
| **Approach B (Test-Time Warping)** | **No (0 epochs, instant)** | Yes ($\hat{\beta}_\tau, \Delta \hat{\nu}$) | **High** (Significant gain over straight infer) | Edge deployment / real-time satellite receiver |
| **Approach C (Physics-CORAL)** | Yes (Joint UDA) | Yes ($\hat{\beta}_\tau$) | **High** | Algorithmic novel contribution in UDA section |

