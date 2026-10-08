# Cross-Domain Inference & Evaluation Workflow

## Directory Layout
- `inference/<output_folder>/<model>/`: Contains SNR test folders (`LS_-10dB`, `LS_0dB`, etc.) with `inferredChannel.mat`.
- `inference/<output_folder>/<model>/synthesized_results.mat`: Per-model synthesized results & PDF plots (`BER_comparison.pdf`, `NMSE_comparison.pdf`, etc.).
- `inference/<output_folder>/syn_<n>/`: Multi-model comparative curves & consolidated metrics across models.
- `inference/<output_folder>/done_infer.md`: Local run summary note.

## Execution
Configure and run:
```powershell
.\inference\cmd_auto_s1_2_3.ps1
```

## Key Parameters in `cmd_auto_s1_2_3.ps1`
- `$trainedDataset`: Source trained dataset name in `single_dataset\` (e.g. `A100_2p18e9_600km_70deg_30kHz`).
- `$outSaveFolderName`: Target output folder name under `inference\` (e.g. `A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz`).
- `$datasetDir`: Target evaluation channel dataset path in `generatedChan\`.
- `$models`: Array of model subfolder names to infer & evaluate.
- `$labels`: Array of legend labels for plots (must match `$models` length).
- `$rerunInfer`: Array of flags matching `$models` length:
  - `0`: Skip Python ONNX inference if `inferredChannel.mat` already exists (fast).
  - `1`: Force rerun Python ONNX inference.
- `$rerunSyn`: Array of flags matching `$models` length:
  - `0`: Skip MATLAB evaluation if `synthesized_results.mat` already exists (fast).
  - `1`: Force rerun MATLAB evaluation and overwrite.
- `$waitForTrain`:
  - `$false`: Run immediately (for manual on-demand execution).
  - `$true`: Poll Git every 20 minutes waiting for `done_train.md` in the source model folder.

> [!IMPORTANT]
> ### Architecture Naming Conventions & Prompt Rules
>
> 1. **Standardization (`std` / `standardize`):**
>    - If requested with **`std`** or **`standardize`**: Target model with **`_standardize`** suffix (e.g. `LI_DnCNN_standardize`, `LS_Attention_standardize`; label adds `std`).
>    - Otherwise: Default to normal model **without `_standardize`** (e.g. `LI_DnCNN`, `LS_Attention`).
>
> 2. **Dataset Scenario Naming Convention (`Source__Target`):**
>    - Target output folder (`$outSaveFolderName`) follows: `<SourceDataset>(_<source_setting>)__<TargetDataset>_<common_or_target_setting>`.
>    - The double underscore **`__`** separates the trained source domain from the target evaluation domain.
>    - The number attached to the channel profile name (e.g. `100` in `A100` or `300` in `DUR300`) represents the **delay spread** (e.g. **100 ns**, **300 ns**).
>    - If no setting is specified immediately after `<SourceDataset>`, both Source and Target share the trailing setting parameters.
>
> 3. **Agent Action Directive:**
>    - When the user asks to infer, evaluate, or synthesize results (e.g. *"run inference and synthesize for dataset X with models Y and Z"*), the agent should directly update `$trainedDataset`, `$outSaveFolderName`, `$datasetDir`, `$models`, `$labels`, `$rerunInfer`, and `$rerunSyn` in `cmd_auto_s1_2_3.ps1` and immediately execute `.\inference\cmd_auto_s1_2_3.ps1`.

## Comparing Multiple Models
To run or compare multiple models on the target domain:
1. Set `$models` and `$labels` in `cmd_auto_s1_2_3.ps1`.
2. Configure `$rerunInfer` and `$rerunSyn` (set to `0` to reuse completed stages).
3. Run:
   ```powershell
   .\inference\cmd_auto_s1_2_3.ps1
   ```
4. Consolidated comparative curves (BER, NMSE, MSE, SSIM) and summary reports are saved to:
   `inference\<output_folder>\syn_<n>\` (auto-increments to `syn_1`, `syn_2`, etc.)

---

## Inference Scripts Comparison & Breakdown

The inference evaluation pipeline consists of three core modular stages:
- **Stage 1 (Python ONNX Inference)**: Generates `inferredChannel.mat` for each SNR point (`LS_-10dB`, `LS_0dB`, ...).
- **Stage 2 (MATLAB Per-Model Metrics)**: Computes MSE, NMSE, SSIM, and BER via `syn_metrics_withBER.m` $\rightarrow$ saves per-model `synthesized_results.mat`.
- **Stage 3 (MATLAB Multi-Model Comparison)**: Compares all models using `syn_syn_compare_multiModels.m` $\rightarrow$ saves consolidated curves in `<output_folder>/syn_<n>/`.

### Summary Comparison Table

| Script | Stage 1: ONNX Inference (Python) | Stage 2: Per-Model Metrics (MATLAB) | Stage 3: Multi-Model Comparison (MATLAB) | Git Polling Trigger | Caching & Rerun Flags | Purpose & Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`cmd_s1_inference.ps1`** | ✅ **Yes** | ❌ No | ❌ No | ❌ No | ❌ No | **Stage 1 Only**: Runs Python batch inference across configured models to generate `inferredChannel.mat`. |
| **`cmd_s2_synMATLAB.ps1`** | ❌ No | ✅ **Yes** | ❌ No | ❌ No | ❌ No | **Stage 2 Only**: Evaluates pre-inferred models in MATLAB individually. Generates per-model `synthesized_results.mat`. |
| **`cmd_s3_compare.ps1`** | ❌ No | ❌ No | ✅ **Yes** | ❌ No | N/A | **Stage 3 Only**: Assumes `synthesized_results.mat` exists across models. Merges them into comparative curves (`syn_1`, `syn_2`, etc.). |
| **`cmd_auto_s1_2_3.ps1`** *(Recommended)* | ✅ **Yes** | ✅ **Yes** | ✅ **Yes** | ✅ Optional (`$waitForTrain`) | ✅ Independent `$rerunInfer` & `$rerunSyn` | **All-in-One Full Pipeline (Stages 1 + 2 + 3)**: Orchestrates the entire workflow from ONNX inference through metrics and comparative plotting. Setting `$rerunInfer = @(0, 0, ...)` skips Stage 1 and runs Stages 2 + 3 directly. |

### Visual Pipeline Overview

```
[Remote Trained Models Ready]
               │
               ▼
┌──────────────────────────────┐
│ Git Polling (Optional)       │ ◄── cmd_auto_s1_2_3.ps1 ($waitForTrain = $true)
│ (done_train.md)              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: Python ONNX Inference                               │ ◄── Standalone: cmd_s1_inference.ps1
│ Generates: <output_folder>/<model>/LS_<snr>/inferredChannel  │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 2: Per-Model Synthesis (syn_metrics_withBER.m)         │ ◄── Standalone: cmd_s2_synMATLAB.ps1
│ Computes MSE, NMSE, SSIM, BER across SNRs                    │
│ Generates: <output_folder>/<model>/synthesized_results.mat   │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 3: Multi-Model Comparison                              │ ◄── Standalone: cmd_s3_compare.ps1
│ (syn_syn_compare_multiModels.m)                              │
│ Generates: <output_folder>/syn_<n>/ comparison curves        │
└──────────────────────────────────────────────────────────────┘
               ▲
               │
    All 3 stages combined with
   caching ($rerunInfer/$rerunSyn):
        cmd_auto_s1_2_3.ps1
   (Setting $rerunInfer=0 runs S2+S3)
```


