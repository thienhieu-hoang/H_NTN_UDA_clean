# CORAL UDA Evaluation & Comparative Workflow

## Directory Layout
- `CORAL/<dataset__dataset>/<model>/<extract_layer>/`: Contains SNR test folders (`LI_-10`, `LS_0`, etc.) with `testChannel_source.mat` and `testChannel_target.mat`.
- `CORAL/<dataset__dataset>/<model>/<extract_layer>/synthesized_results.mat`: Per-layer/block synthesized results (`synthesized_results_target.mat`, `synthesized_results_source.mat`, and `BER_comparison.pdf`, etc.) saved directly inside the layer folder.
- `CORAL/<dataset__dataset>/<model>/syn_<n>/`: Per-model comparative curves across layers/blocks within one model.
- `CORAL/<dataset__dataset>/syn/syn_<n>/`: Multi-model consolidated comparative curves across different models and architectures.
- `CORAL/<dataset__dataset>/done_train.md`: Trigger file indicating remote training completed.

## Execution
Configure and run:
```powershell
.\CORAL\cmd_auto_syn.ps1
```

## Key Parameters in `cmd_auto_syn.ps1`
- `$dataset`: Target UDA dataset scenario folder name formatted as `<SourceDataset>(_<source_setting>)__<TargetDataset>_<common_or_target_setting>`:
  - Separated by a double underscore **`__`** between Source and Target domains.
  - The number directly attached to the channel name (e.g. `100` in `A100` or `DUR100`) denotes the **delay spread** (e.g. **100 ns**).
  - If there is no setting suffix directly after `<SourceDataset>`, both Source and Target share the trailing setting parameters (e.g. in `A100__DUR100_2p18e9_600km_30kHz`, both `A100` and `DUR100` use `2p18e9_600km_30kHz`).
- `$models`: Array of model subfolder names to evaluate.
- `$labels`: Array of base legend labels for plots (must match `$models` length).
- `$extractLayers`: Array of layer/block targets matching `$models` length:
  - `@()` or `$null`: **Auto-discovery mode** — scans and evaluates all subfolders matching `block*` or `layer*`.
  - `@("block3")` or `@("layer1_layer2")`: Evaluates only the specified layer subfolder(s).

> [!IMPORTANT]
> ### Architecture Naming Conventions & Prompt Rules
>
> 1. **Layer vs. Block Subfolder Mapping:**
>    - **`LI_*` models (DnCNN / CNN based)**: Feature folders use **`block*`** (e.g. `block3`, `block2_block3`).
>    - **`LS_*` models (Transformer / Attention based)**: Feature folders use **`layer*`** (e.g. `layer1`, `layer1_layer2`).
>    - **Generic layer prompt mapping**: When requested to evaluate across both model families with generic names (e.g. *"synthesize both at layer 2"* or *"at block 3"*):
>      - Map **`LI_*`** $\rightarrow$ **`block*`** subfolders (`block2_block3` or `block3`).
>      - Map **`LS_*`** $\rightarrow$ **`layer*`** subfolders (`layer1_layer2` or `layer1`).
>      - Or set `$extractLayers = @()` to auto-discover all existing block/layer folders.
>      - *(Note: `cmd_auto_syn.ps1` also includes automatic fallback alias mapping that converts `layer` $\leftrightarrow$ `block` if mismatched).*
>
> 2. **Standardization (`std` / `standardize`):**
>    - If requested with **`std`** or **`standardize`**: Target model with **`_standardize`** suffix (e.g. `LI_DnCNN_standardize`, `LS_Attention_standardize`; label adds `std`).
>    - Otherwise: Default to normal model **without `_standardize`** (e.g. `LI_DnCNN`, `LS_Attention`).
>
> 3. **Projection Head (`pHead` / `projectionHead`):**
>    - If requested with **`pHead`** or **`projectionHead`**: Target model with **`_pHead`** (e.g. `LI_DnCNN_pHead`, `LS_Attention_pHead_standardize`; label adds `pHead`).
>    - Otherwise: Default to normal model **without `_pHead`**.
>
> 4. **Dataset Scenario Naming Convention (`Source__Target`):**
>    - Folders follow the format: `<SourceDataset>(_<source_setting>)__<TargetDataset>_<common_or_target_setting>`.
>    - The double underscore **`__`** separates the source domain from the target domain.
>    - The number attached to the channel profile name (e.g. `100` in `A100` or `DUR100`) represents the **delay spread** (e.g. **100 ns**).
>    - If no setting is specified immediately after `<SourceDataset>`, both Source and Target share the same setting parameters specified at the end (e.g. `A100__DUR100_2p18e9_600km_30kHz` indicates both `A100` and `DUR100` share `2p18e9_600km_30kHz`).
>
> 5. **Agent Action Directive:**
>    - When the user asks to synthesize or evaluate results (e.g. *"synthesize the results of CORAL UDA for dataset X with models Y and Z"*), the agent should directly update `$dataset`, `$models`, `$labels`, and `$extractLayers` in `cmd_auto_syn.ps1` and immediately execute `.\CORAL\cmd_auto_syn.ps1`.

- `$rerun`: Array of flags matching `$models` length:
  - `0`: Skip MATLAB simulation if `synthesized_results.mat` already exists in that layer folder (fast).
  - `1`: Force rerun MATLAB evaluation and overwrite.
- `$waitForTrain`:
  - `$false`: Run immediately (for manual on-demand execution).
  - `$true`: Poll Git every 20 minutes waiting for `done_train.md`.

## Comparative Synthesis (Single Model Across Layers vs. Multi-Model)
To run or compare models across architectures and layers:
1. Configure `$models`, `$labels`, and `$extractLayers` in `cmd_auto_syn.ps1`:
   - **Single model (`$models.Length == 1`)**:
     - Auto-discovers all `block*` or `layer*` subfolders within that model (when `$extractLayers = @(@())`).
     - Generates comparative curves across its different layers/blocks.
     - Saves results locally inside the model folder: `CORAL/<dataset__dataset>/<model>/syn_<n>/` (e.g. `LI_DnCNN/syn_1`, `syn_2`).
   - **Multiple models (`$models.Length > 1`)**:
     - Evaluates all configured models and their respective layers.
     - Generates consolidated comparative curves across all models and benchmarks.
     - Saves results at the dataset level: `CORAL/<dataset__dataset>/syn/syn_<n>/` (e.g. `syn/syn_1`, `syn/syn_2`).
2. Set `$rerun = @(0, ...)` to reuse existing synthesis results without re-running MATLAB simulations.
3. Run:
   ```powershell
   .\CORAL\cmd_auto_syn.ps1
   ```
