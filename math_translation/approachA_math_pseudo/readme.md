# Approach A: Physical Pseudo-Dataset Generation

## 1. Overview & Purpose
The script [`generate_pseudo_dataset.py`](./generate_pseudo_dataset.py) is a **standalone dataset generation tool** (it does **not** train any neural network models).

Its sole objective is to synthetically bridge domain gaps between a Source scenario (e.g., `A100`) and a Target scenario (e.g., `DUR300`) by applying analytical physical channel transformations:
1. **Delay Dilation ($\beta_\tau$)**: Rescales frequency selectivity (delay spread) across subcarriers.
2. **Doppler Phase Modulation ($\Delta\nu$)**: Rotates time-varying phase across OFDM symbols.
3. **Power Scaling / Calibration**: Matches target domain RMS energy levels.
4. **5G NR Pilot Simulation**: Samples true channel at DMRS pilot locations, adds complex AWGN according to each target SNR, and performs 2D linear interpolation ($H_{\text{li}}$).
5. **Standard Formatting**: Exports `.mat` files structured identically to standard NTN datasets (`H_perfect`, `H_li`, `H_ls_pilots`, `pilot_rows`, `pilot_cols`).

---

## 2. Detailed Generation Flow & Scaling Methodology

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   INPUT CHANNELS (TIME-FREQUENCY)                                │
│  Source Ground Truth H_src (132x14, 100 ns, 70°)   Target Reference Pilots H_ls^tgt (88 pilots)  │
└────────────────────────────────┬──────────────────────────────────────────┬──────────────────────┘
                                 │                                          │
                                 │        1. BLIND PARAMETER ESTIMATION     │
                                 │        Autocorrelation on sparse pilots  ▼
                                 │        ► Delay Dilation:   β_τ ≈ 3.30
                                 │        ► Doppler Shift:    Δν  ≈ -2.13 Hz
                                 │        ► Target Level:     RMS_tgt or (μ_tgt, σ_tgt)
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP A: PHYSICAL MATHEMATICAL WARPING (COORDINATE TRANSFORMATION)                                │
│ • Delay Dilation:  Stretches frequency axis by factor β_τ (Bc compressed 3.3x, fast ripples)     │
│ • Doppler Shift:   Applies phase rotation exp(j * 2π * Δν * t) across OFDM symbols               │
│                    H_pseudo_perf = Warp(H_src, β_τ, Δν)                                          │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP B: ENERGY & PATH LOSS SCALING (--scale-mode)                                                │
│ Fixes the ~15-30x amplitude difference between Source and Target simulation tools:               │
│                                                                                                  │
│ • "rms" (Default & Recommended):                                                                 │
│     Calibrates total power to match target domain without altering Rayleigh fading zero-mean:    │
│     H_pseudo = H_pseudo_perf * (RMS_tgt / RMS_src)                                               │
│                                                                                                  │
│ • "standardize" / "std":                                                                         │
│     Aligns mean and standard deviation to target distribution:                                   │
│     H_pseudo = (H_pseudo_perf - μ_src) / σ_src * σ_tgt + μ_tgt                                   │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP C: 5G NR TRANSMISSION & NOISE REALIZATION (Per Target SNR: -10 to +15 dB)                   │
│ 1. Extract true pilot values at DMRS Type 2 Port 1 locations: H_true_pilots (88 REs)             │
│ 2. Calibrate complex AWGN variance to pilot power: σ² = P_pilot / 10^(SNR/10)                    │
│ 3. Add noise to simulate real receiver observation: H_ls_pilots = H_true_pilots + Noise         │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP D: 2D LINEAR INTERPOLATION & METRIC CALCULATION                                             │
│ • Reconstruct initial baseline channel: H_li = Interpolate2D(H_ls_pilots)                        │
│ • Compute baseline NMSE (dB) and SSIM against pseudo ground-truth H_pseudo                       │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STEP E: SAVING STANDARD DATASET                                                                  │
│ Saves matlabNTN.mat under SNR_<snr>dB/ with variables:                                           │
│ H_perfect, H_li, H_ls_pilots, pilot_rows, pilot_cols, beta_tau, delta_nu                         │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Why Energy Rescaling is Required:
1. **Physical Path Loss Discrepancy**: Source `A100` ($70^\circ$, MATLAB) has an RMS power of $\approx 1.7 \times 10^{-8}$, whereas Target `DUR300` ($30^\circ$, OpenNTN) has an RMS power of $\approx 6.1 \times 10^{-10}$ (nearly a $30\times$ difference). Warping changes channel shape and frequency ripples, but does **not** adjust power. Rescaling ensures pseudo signal levels match authentic target levels.
2. **Accurate AWGN Realization**: In Step C, noise power is calibrated against pilot power. Proper scaling ensures realistic receiver SNR and noise floors.
3. **Downstream Neural Network Preprocessing Alignment**:
   - Models trained with `_rms` (e.g. `LI_DnCNN_rms`, `LS_Attention_rms`) should be trained on pseudo datasets generated with `--scale-mode rms`.
   - Models trained with `_standardize` (e.g. `LS_Attention_standardize`) should be trained on pseudo datasets generated with `--scale-mode standardize`.

---

## 3. Dataset Storage Location

The generated pseudo datasets are saved under the central repository directory:
```
generatedChan_mathPseudo/
└── <dataset_name>/                    # e.g., A100_DUR300_LS
    ├── SNR_-10dB/matlabNTN.mat
    ├── SNR_-5dB/matlabNTN.mat
    ├── SNR_0dB/matlabNTN.mat
    ├── SNR_5dB/matlabNTN.mat
    ├── SNR_10dB/matlabNTN.mat
    ├── SNR_15dB/matlabNTN.mat
    ├── note.md                        # Summary metrics, estimated dilation/Doppler & links
    ├── note_source.md                 # Copied source scenario configuration
    ├── note_target.md                 # Copied target scenario configuration
    └── pseudo_dataset_comparison.png  # Qualitative visual comparison
```

---

## 4. Running the Generator

The Python script uses `argparse` command-line argument parsers, meaning **you do not need to modify the Python code**. 

You can configure and trigger dataset generation directly from PowerShell via [`run_py.ps1`](./run_py.ps1):

```powershell
# In PowerShell:
.\math_translation\approachA_math_pseudo\run_py.ps1
```

### Key Parameters:
| Parameter | Description | Example / Default |
| :--- | :--- | :--- |
| `--source-dir` | Path to source channel data | `generatedChan\MATLAB\A100_...` |
| `--target-dir` | Path to target channel data (for pilot reference) | `generatedChan\OpenNTN\DUR300...` |
| `--out-dir` | Destination path for pseudo dataset | `generatedChan_mathPseudo\A100_DUR300_LS` |
| `--scale-mode` | Channel scaling mode (`rms`, `standardize`, `none`) | `rms` |
| `--snr-list` | Target SNR points to generate | `-10 -5 0 5 10 15` |
| `--num-samples` | Limit number of samples (optional for quick tests) | Omit for all samples |
| `--beta-tau` | Manual delay dilation factor override | Omit to auto-estimate from pilots |
| `--delta-nu` | Manual Doppler shift in Hz override | Omit to auto-estimate from pilots |

---

## 5. End-to-End Workflow & Next Steps

Once the pseudo dataset is generated in `generatedChan_mathPseudo/`, proceed with the standard training and evaluation pipeline:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Math Pseudo Generation (This Folder)                                │
│    python generate_pseudo_dataset.py                                   │
│    -> Output: generatedChan_mathPseudo\<pseudo_dataset_name>\          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Single Dataset Training & Benchmark                                 │
│    Train AI models (DnCNN, Attention, etc.) on the pseudo dataset      │
│    -> Output: single_dataset\math_pseudo_*\                            │
│    (e.g., single_dataset\math_pseudo_A100_DUR300_LS\)                  │
│    Run benchmark script: .\single_dataset\cmd_auto_step1_2_syn.ps1     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Cross-Domain Inference to Real Target                               │
│    Infer the pseudo-trained models onto real target channel            │
│    -> Output: inference\math_pseudo_*\                                 │
│    (e.g., inference\math_pseudo_A100_DUR300_LS__DUR300\)               │
│    Run inference pipeline: .\inference\cmd_auto_s1_2_3.ps1             │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Step 1 (Single Dataset Training)**:
   - Train baseline models (e.g., `LI_DnCNN_rms`, `LS_Attention_rms`) on the newly generated dataset in `generatedChan_mathPseudo/<pseudo_dataset_name>/`.
   - Results and trained weights are placed in:
     `single_dataset\math_pseudo_<name>\`
   - Run synthesis with `.\single_dataset\cmd_auto_step1_2_syn.ps1`.

2. **Step 2 (Cross-Domain Target Inference)**:
   - Evaluate model generalization by testing the pseudo-trained weights on the actual target dataset (e.g. `generatedChan/OpenNTN/DUR300...`).
   - Results, BER curves, NMSE, and comparison plots are saved in:
     `inference\math_pseudo_<name>__<target_name>\`
   - Run the 3-stage inference pipeline using `.\inference\cmd_auto_s1_2_3.ps1`.
