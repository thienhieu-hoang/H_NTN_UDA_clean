# cmd_auto_syn.ps1
#
# OVERVIEW:
#   This script is an automated end-to-end pipeline that orchestrates single-dataset evaluations:
#
#   1. Training Wait Loop (Git Polling):
#      Polls Git every 20 minutes (with instant local-first check) looking for a
#      "done_train.md" trigger file inside the trained dataset folder.
#
#   2. Batch Performance Evaluation:
#      Once training completion is detected (or run manually), it loops through configured model
#      subfolders, checking whether synthesize results already exist and respects the $rerun flag.
#      Calls syn_results_withBER in MATLAB to synthesize evaluation metrics (MSE, NMSE, SSIM, BER).
#
#   3. Consolidated Comparative Plotting:
#      Detects model input structures (LI vs LS), gathers their synthesis folders,
#      finds the next incremental results directory index (e.g., syn_1, syn_2...),
#      and triggers syn_syn_results_ in MATLAB to generate consolidated comparison PDF curves.
#
# RUN IN POWERSHELL:
#   .\single_dataset\cmd_auto_syn.ps1
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# Note on available values for $models (in A100_2p18e9_600km_70deg_30kHz):
#   "LI_cGAN",
#   "LI_cGAN_standardize",
#   "LI_DnCNN",
#   "LI_DnCNN_standardize",
#   "LI_DnCNN_Attention",
#   "LI_DnCNN_Attention_standardize",
#   "LI_DnCNN_AxialAttention",
#   "LI_DnCNN_AxialAttention_standardize",
#   "LI_DnCNN_CrossAttention",
#   "LI_DnCNN_CrossAttention_standardize",
#   "LS_Attention",
#   "LS_Attention_standardize",
#   "LS_Attention_AxialAttention",
#   "LS_Attention_AxialAttention_standardize",
#   "LS_Attention_cGAN",
#   "LS_Attention_cGAN_standardize",
#   "LS_Attention_DualDomain",
#   "LS_Attention_DualDomain_standardize",
#   "LS_Attention_ResidualRefine",
#   "LS_Attention_ResidualRefine_standardize",
#   "LS_Attention_UNetRefine",
#   "LS_Attention_UNetRefine_standardize",
#   "LS_DnCNN_Attention"
#
# and corresponding labels:
#   "LI+cGAN",
#   "LI+cGAN std",
#   "LI+DnCNN",
#   "LI+DnCNN std",
#   "LI+DnCNN+Transformer",
#   "LI+DnCNN+Transformer std",
#   "LI+DnCNN+AxialTransformer",
#   "LI+DnCNN+AxialTransformer std",
#   "LI+DnCNN+CrossTransformer",
#   "LI+DnCNN+CrossTransformer std",
#   "LS+Transformer",
#   "LS+Transformer std",
#   "LS+Transformer+AxialTransformer",
#   "LS+Transformer+AxialTransformer std",
#   "LS+Transformer+cGAN",
#   "LS+Transformer+cGAN std",
#   "LS+Transformer+DualDomain",
#   "LS+Transformer+DualDomain std",
#   "LS+Transformer+ResidualTransformer",
#   "LS+Transformer+ResidualTransformer std",
#   "LS+Transformer+UNetTransformer",
#   "LS+Transformer+UNetTransformer std",
#   "LS+Transformer+DnCNN"
# --------------------------------------------------------------------------------------


# Automatically change directory to the folder containing this script
Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " MATLAB Batch Performance Evaluation & Comparative Plotting Loop (PowerShell)"
Write-Output "======================================================================`n"

# 1. Define root folder for trained models and the trained dataset parameter
$modelRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset"
$trainedDataset = "A100_2p18e9_600km_70deg_30kHz"

# 2. List the model subfolders to evaluate
$models = @(
    "LS_Attention_cGAN",
    "LS_Attention_cGAN_standardize"
)

# Corresponding labels/legend names for evaluation
$labels = @(
    "LS+Transformer+cGAN",
    "LS+Transformer+cGAN std"
)

# Rerun flag for each model:
#   0 = Skip MATLAB evaluation if synthesize results already exist (reuse existing)
#   1 = Force rerun MATLAB evaluation and overwrite existing results
$rerun = @(
    0,
    0
)

# Verify list lengths match
if ($models.Length -ne $labels.Length -or $models.Length -ne $rerun.Length) {
    Write-Error "Error: The number of models ($($models.Length)), labels ($($labels.Length)), and rerun flags ($($rerun.Length)) must match!"
    Exit
}

# 3. Git polling configuration for training completion flag (done_train.md)
# Set $waitForTrigger = $false to run immediately without polling for done_train.md
$waitForTrigger = $true
$triggerFile = Join-Path (Join-Path $modelRootDir $trainedDataset) "done_train.md"
$checkIntervalSeconds = 1200  # Poll every 20 minutes

Write-Output "Polling git pull every 20 minutes..."
Write-Output "Looking for trigger file: $triggerFile`n"

while ($true) {
    # Check if trigger file exists locally first or if trigger waiting is disabled
    if (-not $waitForTrigger -or (Test-Path $triggerFile)) {
        if (Test-Path $triggerFile) {
            Write-Output "`n[TRIGGER DETECTED] Found done_train.md inside model folder!"
        }
        else {
            Write-Output "`n[MANUAL RUN] Running evaluation immediately (waitForTrigger = false)..."
        }
        Write-Output "Starting MATLAB batch evaluation loop..."
        
        # Automatically locate official MATLAB CLI launcher (prevents GUI detachment)
        $matlabExe = "matlab"
        if (Test-Path "C:\Program Files\MATLAB\R2025a\bin\matlab.exe") {
            $matlabExe = "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
        }

        # Loop through each model and run the MATLAB evaluation function (syn_results_withBER)
        for ($i = 0; $i -lt $models.Length; $i++) {
            $model = $models[$i]
            $label = $labels[$i]
            $forceRerun = ($rerun[$i] -eq 1) -or ($rerun[$i] -eq $true)
            
            # Construct absolute evaluation folder path
            $evalFolder = Join-Path (Join-Path $modelRootDir $trainedDataset) $model
            
            # Double check if folder exists before invoking MATLAB
            if (-not (Test-Path $evalFolder)) {
                Write-Warning "Skipping model '$model': Path '$evalFolder' does not exist."
                continue
            }
            
            # Determine expected synthesize folder and MAT result file
            $prefix = "LI"
            if ($model.StartsWith("LS_")) {
                $prefix = "LS"
            }
            $synFolder = Join-Path $evalFolder "${prefix}_synthesize"
            $synMatFile = Join-Path $synFolder "synthesized_results.mat"
            $alreadySynthesized = (Test-Path $synFolder) -and (Test-Path $synMatFile)

            Write-Output "------------------------------------------------------------"
            Write-Output "Processing Run $($i + 1)/$($models.Length):"
            Write-Output "  Model Folder : $model"
            Write-Output "  Full Path    : $evalFolder"
            Write-Output "  Plot Label   : $label"
            Write-Output "  Rerun Flag   : $($rerun[$i])"

            # Check if we should skip MATLAB evaluation
            if ($alreadySynthesized -and -not $forceRerun) {
                Write-Output "  Status       : [SKIP] Results already exist at '$synFolder'."
                Write-Output "                 Skipping MATLAB evaluation (rerun = 0)."
                Write-Output "------------------------------------------------------------`n"
                continue
            }

            if ($alreadySynthesized -and $forceRerun) {
                Write-Output "  Status       : [RERUN] Results exist, but rerun = 1. Overwriting..."
            }
            else {
                Write-Output "  Status       : [NEW] Running MATLAB evaluation..."
            }
            Write-Output "------------------------------------------------------------"

            # Invoke MATLAB in headless batch mode, calling the syn_results_withBER function
            $escapedFolder = $evalFolder.Replace('\', '/')
            & $matlabExe -batch "syn_results_withBER('$escapedFolder', '$label')"
            
            if ($LASTEXITCODE -eq 0) {
                Write-Output "Evaluation for '$model' completed successfully.`n"
            }
            else {
                Write-Warning "Evaluation for '$model' failed with exit code $LASTEXITCODE.`n"
            }
        }

        # 4. Generate Consolidated Comparative Plots (syn_syn_results_)
        Write-Output "======================================================================"
        Write-Output " Generating overall comparative plots..."
        Write-Output "======================================================================`n"

        # Construct cell array folders for comparison (pointing to LI_synthesize or LS_synthesize)
        $compFolders = @()
        $compLabels = @()
        for ($i = 0; $i -lt $models.Length; $i++) {
            $model = $models[$i]
            $label = $labels[$i]
            $evalFolder = Join-Path (Join-Path $modelRootDir $trainedDataset) $model
            $prefix = "LI"
            if ($model.StartsWith("LS_")) {
                $prefix = "LS"
            }
            $synFolder = Join-Path $evalFolder "${prefix}_synthesize"
            $synMat = Join-Path $synFolder "synthesized_results.mat"

            if (Test-Path $synMat) {
                $compFolders += $synFolder
                $compLabels += $label
            }
            else {
                Write-Warning "Excluding '$model' from comparative plot: '$synMat' not found."
            }
        }

        if ($compFolders.Length -eq 0) {
            Write-Error "No valid synthesized results found to compare! Skipping overall plot."
            break
        }

        # Incremental output folder search (syn_x) under the dataset's syn/ folder
        $parentPath = Join-Path (Join-Path $modelRootDir $trainedDataset) "syn"
        if (-not (Test-Path $parentPath)) {
            New-Item -ItemType Directory -Force -Path $parentPath | Out-Null
        }

        $maxNum = 0
        $subDirs = Get-ChildItem -Path $parentPath -Directory -Filter "syn*"
        foreach ($dir in $subDirs) {
            if ($dir.Name -match '^syn_?(\d+)$') {
                $num = [int]$Matches[1]
                if ($num -gt $maxNum) {
                    $maxNum = $num
                }
            }
        }
        $nextNum = $maxNum + 1
        $compareOutFolder = Join-Path $parentPath "syn_$nextNum"
        $escapedCompareOut = $compareOutFolder.Replace('\', '/')

        # Build MATLAB cell arrays
        $matlabFoldersCell = "{" + (($compFolders | ForEach-Object { "'$( $_.Replace('\', '/') )'" }) -join ", ") + "}"
        $matlabLabelsCell = "{" + (($compLabels | ForEach-Object { "'$_'" }) -join ", ") + "}"
        $compareCmd = "syn_syn_results_($matlabFoldersCell, $matlabLabelsCell, '$escapedCompareOut')"

        Write-Output "Saving combined comparison to: $compareOutFolder"
        & $matlabExe -batch $compareCmd

        if ($LASTEXITCODE -eq 0) {
            Write-Output "`nOverall comparative plots completed successfully.`n"
        }
        else {
            Write-Warning "`nOverall comparative plots failed with exit code $LASTEXITCODE.`n"
        }

        Write-Output "Done! Stopping poll loop."
        break
    }

    # If not found locally, run git pull to check remote updates
    Write-Output "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Trigger file not found locally. Checking remote with git pull..."
    git pull

    # Check again immediately after pulling
    if (Test-Path $triggerFile) {
        continue  # Loops back to the top to trigger the execution block immediately
    }

    # If still not found, sleep for 20 minutes before checking again
    Write-Output "No trigger file found. Waiting 20 minutes..."
    Start-Sleep -Seconds $checkIntervalSeconds
}
