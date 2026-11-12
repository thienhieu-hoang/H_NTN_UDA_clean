# Approach B: Test-Time Domain Adaptation (Zero-Retraining Warping)

## 1. Overview & Purpose
The script [`test_time_warping_infer.py`](./test_time_warping_infer.py) implements **on-the-fly, test-time domain adaptation (TTA)** using physical coordinate warping.

Unlike Approach A (which requires offline pseudo-dataset synthesis and multi-hour neural network retraining), **Approach B requires zero retraining and generates no offline pseudo datasets**. 

It enables an existing **frozen Source Model** (e.g., `LS_Attention_rms` trained purely on the Source domain `A100` at $100\text{ ns}$ delay spread, $70^\circ$ elevation) to accurately estimate channels on an unseen Target domain (e.g., `DUR300` at $300\text{ ns}$ delay spread, $30^\circ$ elevation, NLoS) in **real-time / on-the-fly**:

1. **Zero Retraining**: Model weights (`.onnx`) remain completely frozen.
2. **On-the-Fly Transform**: Incoming target pilots are warped in real time immediately before model evaluation, and predicted channel grids are warped back immediately after.
3. **Pilot Anchoring**: Measured target pilots are directly substituted/blended into the predicted grid, guaranteeing zero estimation error at pilot locations.
4. **Instant Evaluation**: Directly benchmarks Straight Inference (Source Model applied naively to Target) against Approach B across all SNR points with full NMSE, SSIM, and runtime metrics.

---

## 2. Detailed Workflow & On-the-Fly Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   INCOMING TARGET PILOTS (RUNTIME)                               │
│           H_ls^tgt (88 pilot REs at DMRS Type 2 locations, 300 ns delay spread, Doppler)         │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 │  BLIND PARAMETER ESTIMATION (from target pilots)
                                 ▼
                                 ► Delay Dilation Factor:   β̂_τ ≈ 3.30
                                 ► Bulk Doppler Shift:      Δν̂  ≈ -2.13 Hz
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: INVERSE PILOT COORDINATE WARPING (DISGUISE TARGET PILOTS)                                │
│ • 2D Linear Interpolation over pilot subcarrier grid                                             │
│ • Coordinate Compression in Delay Domain:  τ' = τ * β̂_τ (Bc stretched 3.3x: 300ns → 100ns)      │
│ • Bulk Doppler De-Modulation:              H_disg = H * exp(-j * 2π * Δν̂ * t * T_sym)             │
│ • Resample at standard DM-RS positions:    p_disguised = H_disg[pilot_rows, pilot_cols]          │
│ Result: Disguised pilots match the delay & Doppler statistics of Source Model 1!                │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: PREPROCESSING (NORMALIZE TO MATCH FROZEN MODEL 1)                                        │
│ Auto-detected from model folder name (RMS, Standardize, or Min-Max):                             │
│ • "rms":          x_in = [Re(p), Im(p)] / RMS_p                                                  │
│ • "standardize":  x_in = ([Re(p), Im(p)] - μ_p) / σ_p                                            │
│ • "minmax":       x_in = scaled to [-1.0, +1.0]                                                  │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: FORWARD PASS THROUGH FROZEN SOURCE MODEL 1 (ONNX)                                        │
│ • Frozen inference on disguised inputs: ŷ_source_domain = Model_1(x_in)                          │
│ • Denormalize back to complex channel grid in source domain: Ĥ_src (132 x 14, 100ns-like)        │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: FORWARD CHANNEL GRID WARPING (RETURN TO TARGET DOMAIN)                                   │
│ • Coordinate Dilation in Delay Domain:     h_tgt(τ, t) = (1 / √β̂_τ) * h_src(τ / β̂_τ, t)           │
│   (Delays stretched by 3.3x: 100ns → 300ns, frequency ripples restored to target selectivity)   │
│ • Bulk Doppler Phase Modulation:           Ĥ_warped = Ĥ_tgt * exp(+j * 2π * Δν̂ * t * T_sym)       │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: PILOT ANCHORING (EXACT PILOT FIDELITY)                                                   │
│ Known measured target pilots H_ls^tgt are blended into the predicted grid at pilot REs:          │
│ Ĥ_final[n, r_p, c_p] = (1 - α) * Ĥ_warped[n, r_p, c_p] + α * H_ls^tgt[n, p]                    │
│ Setting α = 1.0 (default) eliminates all residual estimation error at pilot REs.                │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: OUTPUTS & BENCHMARK METRICS                                                              │
│ • Save inferredChannel.mat under LS_<snr>dB/                                                     │
│ • Plot NMSE (dB) and SSIM curves vs Straight Inference: approachB_snr_curves.png                 │
│ • Write comparative markdown summary: results_summary.md                                         │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Comparison of Adaptation Paradigms

| Feature | [Approach A (`approachA_pseudo`)](../approachA_pseudo) | **Approach B (This Folder)** | [Approach C (`approachC_physics_coral`)](../approachC_physics_coral) |
| :--- | :--- | :--- | :--- |
| **Adaptation Level** | Data-Level (Pseudo Dataset) | **Test-Time (Zero-Retraining Warping)** | Feature-Level (Joint UDA Training) |
| **Generates Pseudo Data?**| **Yes** (`generatedChan_mathPseudo/`) | **No (0 MB dataset generated)** | No (uses unlabelled target pilots) |
| **Retrains Neural Net?** | **Yes** (Full training on pseudo data) | **No (Model 1 stays 100% frozen)** | **Yes** (Joint source + target UDA) |
| **Execution Latency** | Offline (Hours for gen + training) | **On-the-fly (Milliseconds per slot)** | Offline (Hours for joint training) |
| **Primary Use-Case** | Long-term model specialization | **Immediate edge/satellite deployment** | Deep feature-space alignment |
| **Pilot Anchoring?** | No (relies on network interpolation) | **Yes (Hard/soft pilot injection)** | Optional post-processing |

---

## 4. Mathematical Mechanics

### 4.1. Step 1: Inverse Coordinate Warping (Target $\rightarrow$ Source Domain)
Target pilots $H_{\text{ls}}^{\text{tgt}}$ have a small coherence bandwidth ($B_c \approx \frac{1}{5 \cdot 300\text{ ns}} \approx 667\text{ kHz}$) and Doppler phase rotation. To make them digestible for Source Model 1 (which was trained on $100\text{ ns}$ delay spread with $B_c \approx 2\text{ MHz}$):

1. **Doppler De-Modulation**:
   $$\tilde{H}_{\text{pilot}}(k, l) = H_{\text{ls}}^{\text{tgt}}(k, l) \cdot \exp\left( -j 2\pi \Delta\hat{\nu} \cdot l \cdot T_{\text{sym}} \right)$$
2. **Delay Compression ($\frac{1}{\hat{\beta}_\tau}$)**:
   The pilot subcarrier profile is mapped via 2D interpolation, transformed to the delay domain $h(\tau, t) = \mathcal{F}_f^{-1}\{\tilde{H}(f, t)\}$, and compressed:
   $$h_{\text{src-like}}(\tau, t) = \sqrt{\hat{\beta}_\tau} \cdot h\left(\hat{\beta}_\tau \cdot \tau, \, t\right)$$
   Transforming back to frequency via FFT produces disguised pilots $p_{\text{disguised}}$ having the wider coherence bandwidth expected by Model 1.

### 4.2. Step 4: Forward Coordinate Warping (Source $\rightarrow$ Target Domain)
After Model 1 outputs the predicted full channel grid $\hat{H}_{\text{src}} \in \mathbb{C}^{132 \times 14}$, it is stretched back to match the true target channel environment:
1. **Delay Dilation ($\hat{\beta}_\tau$)**:
   $$h_{\text{tgt}}(\tau, t) = \frac{1}{\sqrt{\hat{\beta}_\tau}} \cdot h_{\text{src}}\left(\frac{\tau}{\hat{\beta}_\tau}, \, t\right)$$
   $$\hat{H}_{\text{tgt}}(f, t) = \mathcal{F}_\tau\left\{ h_{\text{tgt}}(\tau, t) \right\}$$
2. **Doppler Phase Modulation**:
   $$\hat{H}_{\text{warped}}(k, l) = \hat{H}_{\text{tgt}}(k, l) \cdot \exp\left( +j 2\pi \Delta\hat{\nu} \cdot l \cdot T_{\text{sym}} \right)$$

### 4.3. Step 5: Pilot Anchoring
At pilot subcarrier/symbol positions $(r_p, c_p)$, the true target pilot measurements $H_{\text{ls}}^{\text{tgt}}[n, p]$ are already observed at the receiver. Using a blend factor $\alpha \in [0.0, 1.0]$:
$$\hat{H}_{\text{final}}[n, r_p, c_p] = (1 - \alpha) \hat{H}_{\text{warped}}[n, r_p, c_p] + \alpha H_{\text{ls}}^{\text{tgt}}[n, p]$$
Setting $\alpha = 1.0$ (default) enforces **hard pilot anchoring**, ensuring that known receiver observations are strictly respected without any neural network distortion.

---

## 5. Artifacts & Outputs

All results are saved automatically to:
```
math_translation/results/approachB_test_time/
├── LS_-10dB/
│   └── inferredChannel.mat           # Transposed H_LS_infer, H_straight_infer, NMSE, SSIM
├── LS_-5dB/
│   └── inferredChannel.mat
├── LS_0dB/
│   └── inferredChannel.mat
├── LS_5dB/
│   └── inferredChannel.mat
├── LS_10dB/
│   └── inferredChannel.mat
├── LS_15dB/
│   └── inferredChannel.mat
├── approachB_snr_curves.png          # High-resolution comparative curve plot
└── results_summary.md                # Markdown table with NMSE, SSIM & dB gains
```

### Key Variables in `inferredChannel.mat`:
- `H_LS_infer`: Final adapted channel tensor $(N_{\text{sym}}, N_{\text{sc}}, N_{\text{samples}}) = (14, 132, N)$, structured identically to MATLAB NTN benchmarks.
- `H_straight_infer`: Channel tensor from straight naive inference (no warping).
- `nmse_db`: Final NMSE in dB of Approach B.
- `nmse_straight_db`: Baseline NMSE in dB of Straight Inference.
- `ssim`: Structural similarity index of Approach B.
- `beta_tau`, `delta_nu`: Physical parameters applied during warping.

---

## 6. How to Run

### Option A: Using the Automated PowerShell Runner (Recommended)
You do **not** need to edit Python scripts. Simply run:
```powershell
# From the repository root or approachB_test_time folder:
cd .\math_translation\approachB_test_time\
.\cmd_run_approachB.ps1
```

Inside [`cmd_run_approachB.ps1`](./cmd_run_approachB.ps1), configurable settings include:
```powershell
$modelDir         = "..\..\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_rms"
$targetDatasetDir = "..\..\generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps"
$sourceRefDir     = "..\..\generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz"
$outDir           = "..\results\approachB_test_time"
$pilotAnchorBlend = 1.0     # 1.0 = exact pilot substitution; 0.0 = no anchoring
$numSamples       = "None"  # "None" for all samples, or integer (e.g. 100 for fast smoke test)
```

### Option B: Using the Python CLI
```bash
python test_time_warping_infer.py \
    --model-dir "../../single_dataset/A100_2p18e9_600km_70deg_30kHz/LS_Attention_rms" \
    --target-dataset-dir "../../generatedChan/OpenNTN/DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps" \
    --source-ref-dir "../../generatedChan/MATLAB\A100_2p18e9_600km_70deg_30kHz" \
    --out-dir "../results/approachB_test_time" \
    --pilot-anchor-blend 1.0
```

### CLI Arguments Reference:
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--model-dir` | Path to trained Source Model 1 directory (contains ONNX) | `single_dataset\...\LS_Attention_rms` |
| `--target-dataset-dir` | Path to target channel dataset containing test SNRs | `generatedChan\OpenNTN\DUR300...` |
| `--source-ref-dir` | Path to source channel dataset (for blind shift baseline) | `generatedChan\MATLAB\A100...` |
| `--out-dir` | Output folder for evaluation metrics and `.mat` files | `math_translation\results\approachB_test_time` |
| `--pilot-anchor-blend` | Blend weight $\alpha \in [0.0, 1.0]$ for pilot anchoring | `1.0` (Hard substitution) |
| `--num-samples` | Max number of test samples per SNR (`None` for all) | `None` |
| `--beta-tau` | Manual override for delay dilation factor $\beta_\tau$ | `None` (Blindly estimated from pilots) |
| `--delta-nu` | Manual override for bulk Doppler shift in Hz $\Delta\nu$ | `None` (Blindly estimated from pilots) |

