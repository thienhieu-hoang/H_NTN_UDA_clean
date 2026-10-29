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

## 2. Dataset Storage Location

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
    ├── pseudo_dataset_comparison.png  # Qualitative visual comparison
    └── note.md                        # Summary metrics & estimated dilation/Doppler
```

> [!NOTE]
> For example: `generatedChan_mathPseudo\A100_DUR300_LS\` represents a pseudo dataset generated from source `A100` targeting `DUR300` using LS pilot reference.

---

## 3. Running the Generator

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

## 4. End-to-End Workflow & Next Steps

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

