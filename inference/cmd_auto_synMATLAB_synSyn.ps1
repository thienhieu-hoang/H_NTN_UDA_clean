# cmd_synSyn_metrics_auto.ps1
#
# OVERVIEW:
#   Automated post-inference MATLAB evaluation and multi-model comparison pipeline with Git polling.
#
# WORKFLOW:
#   1. Inference Wait Loop (Git Polling):
#      Polls Git every 20 minutes (with an immediate local check) looking for the "done_infer.md"
#      trigger file inside the specified inference folder ($inferenceRootDir\$inferredPath).
#
#   2. Individual Model Performance Evaluation:
#      Once "done_infer.md" is detected, iterates through each configured model subfolder and runs
#      the MATLAB function "syn_metrics_withBER" in headless batch mode to calculate SNR performance
#      metrics (NMSE, MMSE, SSIM, BER) and save per-model evaluation plots.
#
#   3. Consolidated Multi-Model Comparative Analysis:
#      Automatically detects the next incremental comparison output folder (e.g., syn_1, syn_2, syn_x)
#      under $inferredPath, and executes "syn_syn_compare_multiModels" in MATLAB to plot comparative
#      curves (MSE, NMSE, SSIM, BER) and generate synthesis comparison reports across all listed models.
#
#   4. Completion:
#      Finishes evaluation and comparison plots without deleting "done_infer.md" or pushing to Git.
#
# USAGE (PowerShell):
#   .\cmd_synSyn_metrics_auto.ps1
# --------------------------------------------------------------------------------------

# Automatically change directory to the folder containing this script
Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " MATLAB Batch Performance Evaluation Loop with Git Polling (PowerShell)"
Write-Output "======================================================================`n"

# 1. Define fixed root path for inference runs
$inferenceRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\inference"

# 2. Define the chosen inferred runs directory
$inferredPath = "A100__DUR100_2p18e9_600km_30kHz"

# 3. List the model subfolders to evaluate
$models = @(
    "LI_DnCNN_Attention",
    "LI_DnCNN_Attention_standardize",
    "LI_DnCNN_AxialAttention",
    "LI_DnCNN_AxialAttention_standardize",
    "LI_DnCNN_CrossAttention",
    "LI_DnCNN_CrossAttention_standardize"
)

# 4. List the corresponding legend/plot label names for each model
$labels = @(
    "LI+DnCNN+Transformer Inferred",
    "LI+DnCNN+Transformer(Std) Inferred",
    "LI+DnCNN+AxialTransformer Inferred",
    "LI+DnCNN+AxialTransformer(Std) Inferred",
    "LI+DnCNN+CrossTransformer Inferred",
    "LI+DnCNN+CrossTransformer(Std) Inferred"
)

# Verify list lengths match
if ($models.Length -ne $labels.Length) {
    Write-Error "Error: The number of models ($($models.Length)) does not match the number of labels ($($labels.Length))!"
    Exit
}

# 5. Git polling configuration
$triggerFile = Join-Path (Join-Path $inferenceRootDir $inferredPath) "done_infer.md"
$checkIntervalSeconds = 1200  # Poll every 20 minutes

Write-Output "Polling git pull every 20 minutes..."
Write-Output "Looking for trigger file: $triggerFile`n"

while ($true) {
    # Check if trigger file exists locally first (for immediate manual trigger or if already pulled)
    if (Test-Path $triggerFile) {
        Write-Output "`n[TRIGGER DETECTED] Found done_infer.md inside target folder!"
        Write-Output "Starting MATLAB batch evaluation loop..."

        # Loop through each model and run the MATLAB evaluation function
        for ($i = 0; $i -lt $models.Length; $i++) {
            $model = $models[$i]
            $label = $labels[$i]
            
            # Construct absolute evaluation folder path
            $evalFolder = Join-Path (Join-Path $inferenceRootDir $inferredPath) $model
            
            # Double check if folder exists before invoking MATLAB
            if (-not (Test-Path $evalFolder)) {
                Write-Warning "Skipping model '$model': Path '$evalFolder' does not exist."
                continue
            }
            
            Write-Output "------------------------------------------------------------"
            Write-Output "Processing Run $($i + 1)/$($models.Length):"
            Write-Output "  Model Folder: $model"
            Write-Output "  Full Path   : $evalFolder"
            Write-Output "  Plot Label  : $label"
            Write-Output "------------------------------------------------------------"
            
            # Automatically locate official MATLAB CLI launcher (prevents GUI detachment)
            $matlabExe = "matlab"
            if (Test-Path "C:\Program Files\MATLAB\R2025a\bin\matlab.exe") {
                $matlabExe = "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
            }

            # Invoke MATLAB in headless batch mode, calling the syn_metrics_withBER function
            $escapedFolder = $evalFolder.Replace('\', '/')
            & $matlabExe -batch "syn_metrics_withBER('$escapedFolder', '$label')"
            
            if ($LASTEXITCODE -eq 0) {
                Write-Output "Evaluation for '$model' completed successfully.`n"
            }
            else {
                Write-Warning "Evaluation for '$model' failed with exit code $LASTEXITCODE.`n"
            }
        }

        # 6. Dynamic comparison output folder detection (syn_x)
        $parentPath = Join-Path $inferenceRootDir $inferredPath
        $maxNum = 0
        if (Test-Path $parentPath) {
            $subDirs = Get-ChildItem -Path $parentPath -Directory -Filter "syn_*"
            foreach ($dir in $subDirs) {
                if ($dir.Name -match '^syn_(\d+)$') {
                    $num = [int]$Matches[1]
                    if ($num -gt $maxNum) {
                        $maxNum = $num
                    }
                }
            }
        }
        $nextNum = $maxNum + 1
        $outputFolder = Join-Path $parentPath "syn_$nextNum"
        $escapedOutFolder = $outputFolder.Replace('\', '/')

        Write-Output "------------------------------------------------------------"
        Write-Output "Running Comparison Plots across all Models..."
        Write-Output "Saving results to: $outputFolder"
        Write-Output "------------------------------------------------------------"

        # Build MATLAB cell arrays for folders and labels
        $matlabFoldersCell = "{" + (($models | ForEach-Object { "'$( (Join-Path $parentPath $_).Replace('\', '/') )'" }) -join ", ") + "}"
        $matlabLabelsCell = "{" + (($labels | ForEach-Object { "'$_'" }) -join ", ") + "}"
        $compareCmd = "syn_syn_compare_multiModels($matlabFoldersCell, $matlabLabelsCell, '$escapedOutFolder')"

        # Run overall comparison in MATLAB
        & $matlabExe -batch $compareCmd

        if ($LASTEXITCODE -eq 0) {
            Write-Output "Overall comparison completed successfully.`n"
        }
        else {
            Write-Warning "Overall comparison failed with exit code $LASTEXITCODE.`n"
        }

        Write-Output "`n======================================================================"
        Write-Output " All evaluations and multi-model comparisons completed successfully."
        Write-Output "======================================================================"
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

