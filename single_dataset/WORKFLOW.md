# Single-Dataset Evaluation Workflow

## Directory Layout
- `<dataset>/<model>/`: Contains SNR test folders (`LS_-10`, `LS_0`, etc.) with `testChannel.mat`.
- `<dataset>/<model>/<prefix>_<snr>/results/`: Contains evaluation/testing results for each SNR point (loss observation, `testChannel.mat`, and sample visualization PDFs).
- `<dataset>/<model>/<prefix>_synthesize/`: Per-model synthesized results (`LI_synthesize` or `LS_synthesize`).
- `<dataset>/syn/syn_<n>/`: Multi-model comparative curves & metrics across models.
- `<dataset>/done_train.md`: Trigger file indicating remote training completed.

## Execution
Configure and run:
```powershell
.\single_dataset\cmd_auto_syn.ps1
```

## Key Parameters in `cmd_auto_syn.ps1`
- `$trainedDataset`: Target dataset name (e.g. `A100_2p18e9_600km_70deg_30kHz`).
- `$models`: Array of model subfolder names to evaluate.
- `$labels`: Array of legend labels for plots (must match `$models` length).
- `$rerun`: Array of flags matching `$models` length:
  - `0`: Skip MATLAB simulation if `<prefix>_synthesize\synthesized_results.mat` already exists (fast).
  - `1`: Force rerun MATLAB simulation and overwrite existing results.
- `$waitForTrigger`:
  - `$false`: Run immediately (for manual on-demand execution).
  - `$true`: Poll Git every 20 minutes waiting for `done_train.md`.

> [!IMPORTANT]
> ### Architecture Naming Conventions & Prompt Rules
>
> 1. **Standardization (`std` / `standardize`):**
>    - If requested with **`std`** or **`standardize`**: Target model with **`_standardize`** suffix (e.g. `LI_DnCNN_standardize`, `LS_Attention_standardize`; label adds `std`).
>    - Otherwise: Default to normal model **without `_standardize`** (e.g. `LI_DnCNN`, `LS_Attention`).
>
> 2. **Dataset Scenario Naming Convention:**
>    - Format: `<ChannelProfile><DelaySpread>_<CarrierFrequency>_<Altitude>_<ElevationAngle>_<SubcarrierSpacing>` (e.g. `A100_2p18e9_600km_70deg_30kHz`).
>    - The number attached to the channel profile name (e.g. `100` in `A100` or `DUR100`) represents the **delay spread** (e.g. **100 ns**).
>
> 3. **Agent Action Directive:**
>    - When the user asks to synthesize or evaluate results (e.g. *"synthesize results for single dataset X with models Y and Z"*), the agent should directly update `$trainedDataset`, `$models`, `$labels`, and `$rerun` in `cmd_auto_syn.ps1` and immediately execute `.\single_dataset\cmd_auto_syn.ps1`.

## Comparing Multiple Models
To generate comparative plots across models:
1. Add target model subfolders to `$models` and desired legend names to `$labels` in `cmd_auto_syn.ps1`.
2. Set `$rerun = @(0, 0, ...)` so already synthesized models are skipped without re-running MATLAB.
3. Run:
   ```powershell
   .\single_dataset\cmd_auto_syn.ps1
   ```
4. Consolidated comparative curves (BER, NMSE, MSE, SSIM) and summary tables are saved to:
   `<dataset>\syn\syn_<n>\` (auto-increments to `syn_1`, `syn_2`, etc.)
