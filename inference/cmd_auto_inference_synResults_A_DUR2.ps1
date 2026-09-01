# run_batch_inference_synResults_auto.ps1
#
# OVERVIEW:
#   This script is an automated end-to-end pipeline that orchestrates model evaluation:
#
#   1. Training Wait Loop (Git Polling):
#      Polls Git every 20 minutes (with instant local-first check) looking for a
#      "done_train.md" trigger file inside the trained dataset folder.
#
#   2. Batch ONNX Inference:
#      Once training completion is detected, it runs python ONNX batch inferences
#      for all configured model subfolders sequentially.
#
#   3. Performance Evaluations:
#      Runs the syn_metrics_withBER MATLAB script in headless batch mode for each model run.
#
#   4. Multi-Model Comparisons:
#      Automatically detects the next incremental comparison folder index (e.g. syn_1, syn_2...)
#      and calls syn_syn_compare_multiModels in MATLAB to compile and plot consolidated results.
#
#   5. Git Cleanup:
#      Deletes the trigger file done_train.md, stages the change, and pushes it back to GitHub.
#
# RUN IN POWERSHELL:
#   .\inference\run_batch_inference_synResults_auto.ps1
# --------------------------------------------------------------------------------------

# Automatically change directory to the folder containing this script
Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " Automated ONNX Inference & MATLAB Evaluation Pipeline (PowerShell)"
Write-Output "======================================================================`n"

# 1. Define root folder for trained models and the trained dataset parameter (Source)
$modelRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset"
$trainedDataset = "A100_2p18e9_600km_70deg_30kHz"

# 2. Define (Target) dataset directory (Common for all runs)
$datasetDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps"

# 3. List the models - the names of subfolders
$models = @(
    "LI_cGAN",
    "LI_DnCNN",
    "LI_DnCNN_Attention"
    # "LI_DnCNN_AxialAttention",
    # "LI_DnCNN_CrossAttention",
    # "LS_Attention",
    # "LS_Attention_standardize"
)

# Corresponding labels/legend names for evaluation
$labels = @(
    "LI+cGAN Inferred",
    "LI+DnCNN Inferred",
    "LI+DnCNN+Attention Inferred"
    # "LI+DnCNN+AxialAttention Inferred",
    # "LI+DnCNN+CrossAttention Inferred",
    # "LS+Attention Inferred",
    # "LS+Attention Std Inferred"
)

# 4. Define root folder for outputs and the folder to save results
$outRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\inference"
$outSaveFolderName = "A100__DUR100_2p18e9_600km_30kHz"

# Other common parameters
$numSamples = "None"          # Limit number of samples (or "None" to process all)
$modelType = "auto"           # "LS", "LI", "PRAC", or "auto"
$clipExtrap = "auto"          # "auto", "true", or "false"

# Verify list lengths match
if ($models.Length -ne $labels.Length) {
    Write-Error "Error: The number of models ($($models.Length)) does not match the number of labels ($($labels.Length))!"
    Exit
}

# Create output parent directory if it does not exist
$outParentDir = Join-Path $outRootDir $outSaveFolderName
if (-not (Test-Path $outParentDir)) {
    New-Item -ItemType Directory -Force -Path $outParentDir | Out-Null
}

# 5. Git polling configuration for training completion flag (done_train.md)
$trainTriggerFile = Join-Path (Join-Path $modelRootDir $trainedDataset) "done_train.md"
$checkIntervalSeconds = 1200  # Poll every 20 minutes

Write-Output "Polling git pull every 20 minutes..."
Write-Output "Looking for training completion flag: $trainTriggerFile`n"

while ($true) {
    # Check if trigger file exists locally first
    if (Test-Path $trainTriggerFile) {
        Write-Output "`n[TRIGGER DETECTED] Found done_train.md inside model folder!"
        Write-Output "Starting batch inference..."
        break
    }

    # If not found locally, run git pull to check remote updates
    Write-Output "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - done_train.md not found. Checking remote with git pull..."
    git pull

    # Check again immediately after pulling
    if (Test-Path $trainTriggerFile) {
        Write-Output "`n[TRIGGER DETECTED] Found done_train.md inside model folder!"
        Write-Output "Starting batch inference..."
        break
    }

    # If still not found, sleep for 20 minutes before checking again
    Write-Output "No done_train.md found. Waiting 20 minutes..."
    Start-Sleep -Seconds $checkIntervalSeconds
}

# 6. Loop through each model and run inference
for ($i = 0; $i -lt $models.Length; $i++) {
    $model = $models[$i]
    
    # Resolve full model directory path: $modelRootDir\$trainedDataset\$model
    $modelDir = Join-Path (Join-Path $modelRootDir $trainedDataset) $model
    
    # Check if the resolved model directory exists
    if (-not (Test-Path $modelDir)) {
        Write-Warning "Skipping model '$model': Directory '$modelDir' does not exist."
        continue
    }
    
    # Resolve output directory for this specific model
    $outDir = Join-Path $outParentDir $model
    
    # Determine inference script dynamically (LS models use sequence inference)
    $inferScript = "inference_onnx_grid.py"
    if ($model -like "LS_Attention*" -or $model -like "LS_*") {
        $inferScript = "inference_onnx_lsSequence.py"
    }

    Write-Output "------------------------------------------------------------"
    Write-Output "Processing Run $($i + 1)/$($models.Length):"
    Write-Output "  Model Name   : $model"
    Write-Output "  Infer Script : $inferScript"
    Write-Output "  Model Dir    : $modelDir"
    Write-Output "  Dataset Dir  : $datasetDir"
    Write-Output "  Output Dir   : $outDir"
    Write-Output "------------------------------------------------------------"
    
    # Execute the python inference command with appropriate parser flags
    conda run -n TF_GPU-py3_11 python $inferScript `
        --model-dir $modelDir `
        --dataset-dir $datasetDir `
        --out-dir $outDir `
        --num-samples $numSamples `
        --model-type $modelType `
        --clip-extrap $clipExtrap
        
    if ($LASTEXITCODE -eq 0) {
        Write-Output "Run $($i + 1) completed successfully.`n"
    }
    else {
        Write-Warning "Run $($i + 1) failed with exit code $LASTEXITCODE.`n"
    }
}

# 7. Run MATLAB batch evaluations (sequential, no done_infer.md needed)
Write-Output "`n======================================================================"
Write-Output " Starting MATLAB Batch Performance Evaluations"
Write-Output "======================================================================`n"

for ($i = 0; $i -lt $models.Length; $i++) {
    $model = $models[$i]
    $label = $labels[$i]
    
    # Construct absolute evaluation folder path
    $evalFolder = Join-Path $outParentDir $model
    
    if (-not (Test-Path $evalFolder)) {
        Write-Warning "Skipping evaluation for model '$model': Path '$evalFolder' does not exist."
        continue
    }
    
    Write-Output "------------------------------------------------------------"
    Write-Output "Evaluating Run $($i + 1)/$($models.Length):"
    Write-Output "  Model Folder: $model"
    Write-Output "  Plot Label  : $label"
    Write-Output "------------------------------------------------------------"
    
    $escapedFolder = $evalFolder -replace '\\', '\\'
    matlab -batch "syn_metrics_withBER('$escapedFolder', '$label')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Output "Evaluation for '$model' completed successfully.`n"
    }
    else {
        Write-Warning "Evaluation for '$model' failed with exit code $LASTEXITCODE.`n"
    }
}

# 8. Dynamic comparison output folder detection (syn_x)
$parentPath = $outParentDir
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
$escapedOutFolder = $outputFolder -replace '\\', '\\'

Write-Output "------------------------------------------------------------"
Write-Output "Running Comparison Plots across all Models..."
Write-Output "Saving results to: $outputFolder"
Write-Output "------------------------------------------------------------"

# Build MATLAB cell arrays for folders and labels
$matlabFoldersCell = "{" + (($models | ForEach-Object { "'$( (Join-Path $parentPath $_) -replace '\\', '\\' )'" }) -join ", ") + "}"
$matlabLabelsCell = "{" + (($labels | ForEach-Object { "'$_'" }) -join ", ") + "}"
$compareCmd = "syn_syn_compare_multiModels($matlabFoldersCell, $matlabLabelsCell, '$escapedOutFolder')"

# Run overall comparison in MATLAB
matlab -batch $compareCmd

if ($LASTEXITCODE -eq 0) {
    Write-Output "Overall comparison completed successfully.`n"
}
else {
    Write-Warning "Overall comparison failed with exit code $LASTEXITCODE.`n"
}

# 9. Create done_infer.md with run details in the output folder and git push it
$inferTriggerFile = Join-Path $outParentDir "done_infer.md"
Write-Output "`nCreating trigger file done_infer.md at: $inferTriggerFile"

$modelsStr = ($models | ForEach-Object { "- $_" }) -join "`n"
$labelsStr = ($labels | ForEach-Object { "- $_" }) -join "`n"

$fileContent = @"
Inference completed successfully on $(Get-Date)

Folder:
$outSaveFolderName

Models:
$modelsStr

Labels:
$labelsStr
"@

Set-Content -Force -Path $inferTriggerFile -Value $fileContent

git add $inferTriggerFile
git commit -m "Batch inference completed, trigger file done_infer.md created"
git push

Write-Output "`n======================================================================"
Write-Output " All automated inference and evaluation runs completed successfully."
Write-Output "======================================================================"
