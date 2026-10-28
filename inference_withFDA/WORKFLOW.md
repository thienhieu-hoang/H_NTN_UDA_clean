# Test-Time FDA Cross-Domain Inference Workflow

This document details the complete end-to-end workflow, architectural design, and execution instructions for **Test-Time Fourier Domain Adaptation (FDA)** channel estimation within `inference_withFDA/`.

---

## 1. High-Level Architecture

Unlike training-time FDA (which requires generating `.mat` pseudo-datasets offline and retraining neural networks), **Test-Time FDA operates directly during inference on models trained only on the source domain**.

```
                      ┌────────────────────────────────────────────────────────┐
                      │              Source Model (e.g., TDL-A 100ns)          │
                      │         Trained ONNX Model (LI+DnCNN / LS+Attention)   │
                      └──────────────────────────┬─────────────────────────────┘
                                                 │
                                                 ▼
┌──────────────────────┐         ┌───────────────────────────────┐
│ Target Test Channel  │         │ Precomputed Source Prototype  │
│  (e.g., DUR 300ns)   │         │ Delay-Doppler Spectrum A_src  │
└──────────┬───────────┘         └───────────────┬───────────────┘
           │                                     │
           ▼                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Python Batch Inference with Test-Time FDA (inference_onnx_with_fda) │
├──────────────────────────────────────────────────────────────────────────────┤
│ 1. Pre-FDA Input Translation:                                                │
│    X_trans = FDA( X_tgt <-- A_src )                                          │
│    • Low-frequency amplitude aligned to source; protects model weights.      │
│                                                                              │
│ 2. Normalization:                                                            │
│    Auto-detected from model logs: RMS | Standardize | Min-Max                │
│                                                                              │
│ 3. ONNX Model Inference:                                                     │
│    • Sequence Models (LS+Attention): Inferred on 88 pilot tokens             │
│    • Grid Models (LI+DnCNN): Inferred on 132x14 interpolated grid           │
│                                                                              │
│ 4. Post-FDA Target Style Restoration:                                        │
│    Y_restored = FDA( Y_model <-- X_tgt )                                     │
│    • Re-injects authentic target Doppler/delay amplitude into output.        │
│                                                                              │
│ 5. Pilot Consistency Projection (PCP):                                       │
│    Y_final(pilots) = (1 - α) * Y_restored(pilots) + α * H_LS(pilots)         │
│                                                                              │
│ Output: inferredChannel.mat + comparison_real_and_magnitude.png/pdf          │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: MATLAB Performance Evaluation (syn_metrics_withBER.m)               │
├──────────────────────────────────────────────────────────────────────────────┤
│ • 5G NR PDSCH Transmission, Equalization, and 16-QAM Demodulation            │
│ • Evaluates BER, NMSE (dB), MSE, SSIM vs. Ground Truth across SNR points     │
│ • Benchmark comparison against LS+Linear Interpolation (H_li) and MMSE       │
│                                                                              │
│ Output: synthesized_results.mat + metrics_vs_SNR.png/pdf                     │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Multi-Model Consolidated Comparison (syn_syn_compare_multiModels.m) │
├──────────────────────────────────────────────────────────────────────────────┤
│ • Consolidates curves across multiple models (e.g. LI+DnCNN vs LS+Attention) │
│ • Automatically creates incremental comparison folders: syn/syn_1, syn_2...  │
│                                                                              │
│ Output: BER_comparison.png/pdf, NMSE_comparison.png/pdf                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure & File Roles

```
inference_withFDA/
│
├── cmd_auto_s1_2_3.ps1              # Master PowerShell automation script (Stages 1 -> 2 -> 3)
│
├── fda_window_utils.py              # Core FDA mathematical engine:
│                                    #   • 2D Delay-Doppler FFT / IFFT transforms
│                                    #   • Hard & Smooth Tukey outward tapering windows
│                                    #   • Prototype source spectrum computation
│                                    #   • Pilot Consistency Projection (PCP)
│
├── inference_onnx_with_fda.py       # Unified ONNX inference script:
│                                    #   • Supports both Grid (LI) and Sequence (LS) models
│                                    #   • Pre-FDA, Post-FDA, and PCP integration
│                                    #   • Normalization auto-detection (RMS/Std/MinMax)
│                                    #   • Generates comparison plots and info.md
│
├── syn_metrics_withBER.m            # Stage 2: MATLAB BER and NMSE computation
├── syn_syn_compare_multiModels.m    # Stage 3: MATLAB multi-model comparison curves
│
├── WORKFLOW.md                      # This workflow documentation file
├── FDA_INFERENCE_WORKFLOW_GUIDE.md  # Detailed mathematical & theoretical recommendations
└── FDA_Smooth_Window_Python_Guide.md# Numerical specification of smooth tapering windows
```

---

## 3. Step-by-Step Execution Pipeline

### Stage 1: Batch ONNX Inference (`inference_onnx_with_fda.py`)

For each model and across all SNR subfolders (`LS_-10dB` to `LS_20dB`):

1. **Model Architecture Detection**:
   - Identifies whether the model expects pilot sequence tokens (e.g. `LS_Attention`, shape `(N, 88, 2)`) or a 2D interpolated grid (e.g. `LI_DnCNN`, shape `(N, 132, 14, 2)`).
2. **Source Prototype Extraction**:
   - If Pre-FDA is enabled, computes or loads the mean Delay-Doppler amplitude prototype $\bar{A}_{\text{src}}(\tau, \nu)$ from the source dataset.
3. **Step 1: Pre-FDA Input Translation**:
   - Replaces the target input's central low-frequency Delay-Doppler amplitude with the source prototype using the chosen window.
   - For `LS_Attention`: translates the 2D grid reconstructed from pilots, then re-samples the 88 adapted pilot tokens.
   - For `LI_DnCNN`: translates the 2D linear interpolated channel grid $H_{\text{li}}$.
4. **Step 2: Normalization Scaling**:
   - Auto-detects training normalization from `final_epoch.txt` (`rms`, `standardize`, or `minmax`) and normalizes input samples.
5. **Step 3: ONNX Inference**:
   - Executes the ONNX session (`ort.InferenceSession`) to obtain predictions $\hat{Y}_{\text{pred}}$.
6. **Step 4: Denormalization**:
   - Reverses normalization back to true complex channel units.
7. **Step 5: Post-FDA Inverse Style Restoration**:
   - Re-injects the authentic target low-frequency Delay-Doppler amplitude from $H_{\text{li}}$ into $\hat{Y}_{\text{pred}}$:
     $$\mathcal{F}(\hat{Y}_{\text{restored}}) = \left( (1 - M) \odot A(\hat{Y}_{\text{pred}}) + M \odot A(H_{\text{li}}) \right) e^{j \Phi(\hat{Y}_{\text{pred}})}$$
8. **Step 6: Pilot Consistency Projection (PCP)**:
   - Soft-blends known pilot subcarriers $(p_r, p_c)$ with the measured LS channel:
     $$\hat{Y}_{\text{final}}[i, p_r, p_c] = (1 - \alpha) \hat{Y}_{\text{restored}}[i, p_r, p_c] + \alpha H_{\text{LS}}[i, :]$$
9. **Step 7: Output Artifacts**:
   - Writes `inferredChannel.mat` containing `H_infer`, `H_perfect`, `H_li`, `H_ls_pilots`, `pilot_rows`, `pilot_cols`, `SNRdB`.
   - Generates side-by-side Real and Magnitude comparison figures: `comparison_real_and_magnitude.png` and `.pdf`.
   - Writes `info.md` detailing parameters and per-SNR NMSE/MMSE/SSIM.

---

### Stage 2: MATLAB Performance Evaluation (`syn_metrics_withBER.m`)

1. Scans each model folder for `inferredChannel.mat` across all SNR points.
2. For each sample:
   - Simulates 5G NR PDSCH OFDM transmission through the ground truth channel.
   - Applies MMSE equalization using the inferred channel $\hat{Y}_{\text{final}}$.
   - Demodulates 16-QAM symbols and computes the Bit Error Rate (BER).
   - Computes NMSE, MSE, and complex SSIM.
3. Generates MMSE benchmark curves and Linear Interpolation baseline curves.
4. Saves `synthesized_results.mat` and `synthesized_results.json` in the model folder.

---

### Stage 3: Multi-Model Consolidated Comparison (`syn_syn_compare_multiModels.m`)

1. Gathers `synthesized_results.mat` from all evaluated models (e.g. `LI_DnCNN_rms` and `LS_Attention_rms`).
2. Auto-detects the next incremental synthesis directory (`syn/syn_1/`, `syn/syn_2/`, ...).
3. Produces publication-quality consolidated figures:
   - `BER_comparison.png` & `BER_comparison.pdf`
   - `NMSE_comparison.png` & `NMSE_comparison.pdf`
   - `MSE_comparison.png` & `MSE_comparison.pdf`
   - `SSIM_comparison.png` & `SSIM_comparison.pdf`
4. Writes `comparison_summary.md` and `synthesized_results.json`.

---

## 4. Configuration Guide (`cmd_auto_s1_2_3.ps1`)

All settings are controlled in the parameter block at the top of [`cmd_auto_s1_2_3.ps1`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN\Hest_NTN_UDA_clean\inference_withFDA\cmd_auto_s1_2_3.ps1):

### Primary Paths & Datasets
| Parameter | Description | Example |
| :--- | :--- | :--- |
| `$modelRootDir` | Root folder where source-trained models reside | `.../single_dataset` |
| `$trainedDataset` | Source dataset folder name | `"A100_2p18e9_600km_70deg_30kHz"` |
| `$outRootDir` | Destination folder for FDA inference | `.../inference_withFDA` |
| `$outSaveFolderName` | Subfolder name for this transfer scenario | `"A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz"` |
| `$datasetDir` | Target dataset directory in `generatedChan/` | `.../DUR300nsFix_...` |
| `$sourceRefPath` | Source dataset directory to extract reference spectrum | `.../sampleWiseDoppler_...A100_...` |

### FDA Adaptation Modes (`$fdaStage`)
| Setting | What it does | Recommended For |
| :--- | :--- | :--- |
| `"dual_pcp"` ⭐ | **Pre-FDA + Post-FDA + Pilot Consistency Projection** | **Best overall performance (Default)** |
| `"dual"` | Pre-FDA on input + Post-FDA on output | Good when pilots have high noise |
| `"post"` | Post-FDA style restoration only (Raw input fed to model) | Fast ablation testing |
| `"pre"` | Pre-FDA input translation only (No post-processing) | Evaluating input domain alignment |
| `"none"` | Standard baseline inference (No FDA applied) | Baseline benchmark |

### Window Configuration (`$windowType`)
| Setting | Window Mask Behavior |
| :--- | :--- |
| `"tukey_outward"` ⭐ | **Smooth outward raised-cosine taper.** Keeps core $13 \times w_w$ at $1.0$, decays $1.0 \to 0.0$ outward over `$taperH` and `$taperW` elements. Eliminates Gibbs ringing. |
| `"hard"` | Legacy binary boxcar window. Sharp step edge at boundary ($13 \times w_w$). |
| `"tukey_inward"` | Smooth inward raised-cosine taper inside nominal box. |
| `"supergaussian"` | Flat-top elliptical super-Gaussian taper. |

### Window Dimensions & Tapers
- `$winH = 13`: Nominal window height along Delay axis (subcarriers).
- `$winW = 3` (or `5`): Nominal window width along Doppler axis (OFDM symbols).
- `$taperH = 2`: Number of transition rows along Delay (weights: $[0.75, 0.25]$).
- `$taperW = 1`: Number of transition columns along Doppler (weight: $[0.50]$).
- `$pcpAlpha = 0.5`: Pilot consistency blending weight ($0.0 = \text{pure model}$, $1.0 = \text{hard LS pilot overwrite}$).

### Execution Controls
- `$models`: Array of model subfolder names (e.g. `@("LI_DnCNN_rms", "LS_Attention_rms")`).
- `$labels`: Array of legend labels for plots (e.g. `@("LI+DnCNN RMS FDA", "LS+Transformer RMS FDA")`).
- `$rerunInfer`: `@(1, 1)` to force re-running Python inference; `@(0, 0)` to skip if `inferredChannel.mat` exists.
- `$rerunSyn`: `@(1, 1)` to force re-running MATLAB evaluation; `@(0, 0)` to skip if `synthesized_results.mat` exists.
- `$numSamples`: Set to `64` for rapid validation, or `$null` to process the entire dataset.

---

## 5. Quick-Start Commands

### Run Full Pipeline via PowerShell (Stages 1, 2, and 3):
```powershell
.\inference_withFDA\cmd_auto_s1_2_3.ps1
```

### Run Standalone Python Test-Time FDA Inference:
```powershell
conda run -n TF_GPU-py3_11 python .\inference_withFDA\inference_onnx_with_fda.py `
    --model-dir "single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_rms" `
    --dataset-dir "generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps" `
    --out-dir "inference_withFDA\A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz\LS_Attention_rms" `
    --source-ref-path "generatedChan\MATLAB\sampleWiseDoppler_wGeometry_A100_2p18e9_600km_70deg_30kHz" `
    --fda-stage "dual_pcp" `
    --window-type "tukey_outward" `
    --win-h 13 --win-w 3 --taper-h 2 --taper-w 1 `
    --pcp-alpha 0.5 `
    --num-samples 64
```
