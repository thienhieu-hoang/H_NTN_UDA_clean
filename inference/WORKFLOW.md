# Cross-Domain Inference & Evaluation Workflow

## Directory Layout
- `inference/<output_folder>/<model>/`: Contains SNR test folders (`LS_-10dB`, `LS_0dB`, etc.) with `inferredChannel.mat`.
- `inference/<output_folder>/<model>/synthesized_results.mat`: Per-model synthesized results & PDF plots (`BER_comparison.pdf`, `NMSE_comparison.pdf`, etc.).
- `inference/<output_folder>/syn_<n>/`: Multi-model comparative curves & consolidated metrics across models.
- `inference/<output_folder>/done_infer.md`: Local run summary note.

## Execution
Configure and run:
```powershell
.\inference\cmd_auto_syn.ps1
```

## Key Parameters in `cmd_auto_syn.ps1`
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
>    - When the user asks to infer, evaluate, or synthesize results (e.g. *"run inference and synthesize for dataset X with models Y and Z"*), the agent should directly update `$trainedDataset`, `$outSaveFolderName`, `$datasetDir`, `$models`, `$labels`, `$rerunInfer`, and `$rerunSyn` in `cmd_auto_syn.ps1` and immediately execute `.\inference\cmd_auto_syn.ps1`.

## Comparing Multiple Models
To run or compare multiple models on the target domain:
1. Set `$models` and `$labels` in `cmd_auto_syn.ps1`.
2. Configure `$rerunInfer` and `$rerunSyn` (set to `0` to reuse completed stages).
3. Run:
   ```powershell
   .\inference\cmd_auto_syn.ps1
   ```
4. Consolidated comparative curves (BER, NMSE, MSE, SSIM) and summary reports are saved to:
   `inference\<output_folder>\syn_<n>\` (auto-increments to `syn_1`, `syn_2`, etc.)
