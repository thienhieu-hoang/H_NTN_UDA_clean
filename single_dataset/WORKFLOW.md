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

---

## Evaluation Scripts Comparison & Breakdown

Below is a breakdown of the differences between the evaluation scripts in this directory and how they combine individual stages:

| Script | Git Poll Wait (`done_train.md`) | Per-Model Eval (`syn_results_withBER.m`) | Multi-Model Comparison (`syn_syn_results_.m`) | Skip/Rerun Cache Control (`$rerun`) | Purpose & Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`cmd_syn_metrics.ps1`** | ❌ No | ✅ **Yes** | ❌ No | ❌ No (runs all listed) | **Step 1 Only**: Generates/updates `<prefix>_synthesize/synthesized_results.mat` for each model independently. |
| **`cmd_synSyn_results.ps1`** | ❌ No | ❌ No (requires existing `.mat`) | ✅ **Yes** | N/A | **Step 2 Only**: Assumes `.mat` files already exist. Reads all selected models and produces combined comparison plots under `<dataset>/syn/syn_<n>/`. |
| **`cmd_auto_syn.ps1`** *(Recommended)* | ✅ Optional (`$waitForTrigger`) | ✅ **Yes** | ✅ **Yes** | ✅ **Yes** (`$rerun` array) | **All-in-One Full Pipeline**: Combines Step 1 + Step 2 with smart caching (`$rerun`) and optional trigger polling (`$waitForTrigger = $true/$false`). Safely skips already-calculated models to quickly produce comparison plots. |
| **`cmd_auto_syn_metrics_synSyn.ps1`** | ✅ Mandatory ($true) | ✅ **Yes** | ✅ **Yes** | ❌ No (always reruns) | **Older/Legacy automated pipeline**: Similar to `cmd_auto_syn.ps1` but lacks `$rerun` caching flag and `$waitForTrigger` toggle (always blocks on `done_train.md`). |
| **`cmd_auto_syn_metrics_synSyn_all.ps1`** | N/A (batch runner) | N/A | N/A | N/A | **Batch meta-runner**: Sequentially executes `cmd_auto_syn_metrics_synSyn1.ps1`, `cmd_auto_syn_metrics_synSyn2.ps1`, and `cmd_auto_syn_metrics_synSyn3.ps1` in series. |

### Visual Pipeline Overview

```
[Remote Training Finishes]
            │
            ▼
┌─────────────────────────┐
│ Git Polling Trigger     │ ◄── Enabled in cmd_auto_syn.ps1 (optional) & cmd_auto_syn_metrics_synSyn.ps1
│ (done_train.md)         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│ Step 1: Per-Model Synthesis (syn_results_withBER.m)     │ ◄── Standalone: cmd_syn_metrics.ps1
│ Computes MSE, NMSE, SSIM, BER across SNRs               │
│ Output: <dataset>/<model>/<prefix>_synthesize/          │
│         └── synthesized_results.mat                     │
└───────────┬─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Multi-Model Comparison (syn_syn_results_.m)     │ ◄── Standalone: cmd_synSyn_results.ps1
│ Merges all models into unified comparison curves        │
│ Output: <dataset>/syn/syn_<n>/                          │
│         ├── BER_comparison.pdf                          │
│         ├── NMSE_comparison.pdf, MSE, SSIM...           │
└─────────────────────────────────────────────────────────┘
            ▲
            │
   Combined together with
 caching ($rerun) in:
   cmd_auto_syn.ps1
```

