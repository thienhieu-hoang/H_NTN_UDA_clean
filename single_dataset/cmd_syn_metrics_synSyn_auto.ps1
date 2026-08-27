# cmd_syn_metrics_synSyn_auto.ps1
#
# OVERVIEW:
#   This script is an automated end-to-end pipeline that orchestrates single-dataset evaluations:
#
#   1. Training Wait Loop (Git Polling):
#      Polls Git every 20 minutes (with instant local-first check) looking for a
#      "done_train.md" trigger file inside the trained dataset folder.
#
#   2. Batch Performance Evaluation:
#      Once training completion is detected, it loops through configured model subfolders
#      and calls syn_results_withBER in MATLAB to synthesize evaluation metrics
#      (MSE, NMSE, SSIM, BER) across all SNR points.
#
#   3. Consolidated Comparative Plotting:
#      Detects model input structures (LI vs LS), auto-resolves their synthesis folders,
#      finds the next incremental results directory index (e.g., syn_1, syn_2...),
#      and triggers syn_syn_results_ in MATLAB to generate consolidated comparison PDF curves.
#
#   4. Git Cleanup:
#      Deletes the trigger file done_train.md, stages the change, and pushes it back to GitHub.
#
# RUN IN POWERSHELL:
#   .\single_dataset\cmd_syn_metrics_synSyn_auto.ps1
# --------------------------------------------------------------------------------------

# Automatically change directory to the folder containing this script
Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " MATLAB Batch Performance Evaluation & Comparative Plotting Loop (PowerShell)"
Write-Output "======================================================================`n"

# 1. Define root folder for trained models and the trained dataset parameter
$modelRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset"
$trainedDataset = "DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps"

# 2. List the model subfolders to evaluate
$models = @(
    "LS_Attention_copy",
    "LI_DnCNN_CrossAttention_standardize_copy"
)

# Corresponding labels/legend names for evaluation
$labels = @(
    "LS+Attention",
    "LI+DnCNN+CrossAttention"
)

# Verify list lengths match
if ($models.Length -ne $labels.Length) {
    Write-Error "Error: The number of models ($($models.Length)) does not match the number of labels ($($labels.Length))!"
    Exit
}

# 3. Git polling configuration for training completion flag (done_train.md)
$triggerFile = Join-Path (Join-Path $modelRootDir $trainedDataset) "done_train.md"
$checkIntervalSeconds = 1200  # Poll every 20 minutes

Write-Output "Polling git pull every 20 minutes..."
Write-Output "Looking for trigger file: $triggerFile`n"

while ($true) {
    # Check if trigger file exists locally first (for immediate manual trigger or if already pulled)
    if (Test-Path $triggerFile) {
        Write-Output "`n[TRIGGER DETECTED] Found done_train.md inside model folder!"
        Write-Output "Starting MATLAB batch evaluation loop..."

        # Loop through each model and run the MATLAB evaluation function (syn_results_withBER)
        for ($i = 0; $i -lt $models.Length; $i++) {
            $model = $models[$i]
            $label = $labels[$i]
            
            # Construct absolute evaluation folder path
            $evalFolder = Join-Path (Join-Path $modelRootDir $trainedDataset) $model
            
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
            
            # Invoke MATLAB in headless batch mode, calling the syn_results_withBER function
            $escapedFolder = $evalFolder -replace '\\', '\\'
            matlab -batch "syn_results_withBER('$escapedFolder', '$label')"
            
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
        for ($i = 0; $i -lt $models.Length; $i++) {
            $model = $models[$i]
            $evalFolder = Join-Path (Join-Path $modelRootDir $trainedDataset) $model
            $prefix = "LI"
            if ($model.StartsWith("LS_")) {
                $prefix = "LS"
            }
            $compFolders += Join-Path $evalFolder "${prefix}_synthesize"
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
        $escapedCompareOut = $compareOutFolder -replace '\\', '\\'

        # Build MATLAB cell arrays
        $matlabFoldersCell = "{" + (($compFolders | ForEach-Object { "'$( $_ -replace '\\', '\\' )'" }) -join ", ") + "}"
        $matlabLabelsCell = "{" + (($labels | ForEach-Object { "'$_'" }) -join ", ") + "}"
        $compareCmd = "syn_syn_results_($matlabFoldersCell, $matlabLabelsCell, '$escapedCompareOut')"

        Write-Output "Saving combined comparison to: $compareOutFolder"
        matlab -batch $compareCmd

        if ($LASTEXITCODE -eq 0) {
            Write-Output "`nOverall comparative plots completed successfully.`n"
        }
        else {
            Write-Warning "`nOverall comparative plots failed with exit code $LASTEXITCODE.`n"
        }

        # Clean up training completion trigger file to prevent running again on next loop
        Write-Output "Cleaning up trigger file done_train.md..."
        Remove-Item $triggerFile
        git add $triggerFile
        git commit -m "Local single_dataset batch evaluations and comparisons completed"
        git push
        
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
