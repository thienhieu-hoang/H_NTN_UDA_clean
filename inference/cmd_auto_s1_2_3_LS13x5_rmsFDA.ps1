# cmd_auto_s1_2_3_LS13x5_rmsFDA.ps1
#
# OVERVIEW:
#   Automated end-to-end inference and evaluation pipeline for rms 13x5 FDA models
#
# RUN IN POWERSHELL:
#   .\inference\cmd_auto_s1_2_3_LS13x5_rmsFDA.ps1
# --------------------------------------------------------------------------------------

# Automatically change directory to the folder containing this script
Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " Automated ONNX Inference & MATLAB Evaluation Pipeline (PowerShell)"
Write-Output " Scenario: rms 13x5 FDA Models"
Write-Output "======================================================================`n"

# 1. Define root folder for trained models and the trained dataset parameter (Source)
$modelRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset"
$trainedDataset = "pseudo_A100Perfect_DUR300LSAttention_rms_13x5"

# 2. Define root folder for outputs and the folder to save results
$outRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\inference"
$outSaveFolderName = "LSpseudo13x5_A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz"

# 3. Define (Target) dataset directory (Common for all runs)
$datasetDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps"

# 4. List the models - the names of subfolders
$models = @(
    "LS_Attention_rmsFDA_minmaxReTrain",
    "LS_Attention_rmsFDA_rmsReTrain",
    "LS_Attention_rmsFDA_stdReTrain"
)

# Corresponding labels/legend names for evaluation
$labels = @(
    "LS+Transformer rmsFDA minmaxPreProcess inferred",
    "LS+Transformer rmsFDA RMSPreProcess inferred",
    "LS+Transformer rmsFDA stdPreProcess inferred"
)

# Rerun flags (must match $models length):
#   $rerunInfer: 0 = Skip Python inference if inferredChannel.mat exists, 1 = Force rerun inference
#   $rerunSyn  : 0 = Skip MATLAB evaluation if synthesized_results.mat exists, 1 = Force rerun MATLAB
$rerunInfer = @(
    1,
    1,
    1
)

$rerunSyn = @(
    1,
    1,
    1
)

# Verify list lengths match
if ($models.Length -ne $labels.Length -or $models.Length -ne $rerunInfer.Length -or $models.Length -ne $rerunSyn.Length) {
    Write-Error "Error: The number of models ($($models.Length)), labels ($($labels.Length)), rerunInfer ($($rerunInfer.Length)), and rerunSyn ($($rerunSyn.Length)) must match!"
    Exit
}

# Other common parameters
$numSamples = "None"          # Limit number of samples (or "None" to process all)
$modelType = "auto"           # "LS", "LI", "PRAC", or "auto"
$clipExtrap = "auto"          # "auto", "true", or "false"

# Create output parent directory if it does not exist
$outParentDir = Join-Path $outRootDir $outSaveFolderName
if (-not (Test-Path $outParentDir)) {
    New-Item -ItemType Directory -Force -Path $outParentDir | Out-Null
}

# 5. Remote training completion polling configuration
$waitForTrain = $false
$trainTriggerFile = Join-Path (Join-Path $modelRootDir $trainedDataset) "done_train.md"
$checkIntervalSeconds = 1200

if ($waitForTrain) {
    Write-Output "Polling git pull every 20 minutes..."
    Write-Output "Looking for training completion flag: $trainTriggerFile`n"

    while ($true) {
        if (Test-Path $trainTriggerFile) {
            Write-Output "Found trigger file '$trainTriggerFile' locally. Starting batch evaluation!`n"
            break
        }

        git pull
        if (Test-Path $trainTriggerFile) {
            Write-Output "Trigger file found after git pull: $trainTriggerFile. Starting batch evaluation!`n"
            break
        }

        Write-Output "No done_train.md found. Waiting 20 minutes..."
        Start-Sleep -Seconds $checkIntervalSeconds
    }
}

# 6. Stage 1: Batch ONNX Inference
Write-Output "`n======================================================================"
Write-Output " Starting Batch ONNX Inference"
Write-Output "======================================================================`n"

for ($i = 0; $i -lt $models.Length; $i++) {
    $model = $models[$i]
    $forceInfer = ($rerunInfer[$i] -eq 1) -or ($rerunInfer[$i] -eq $true)
    
    $modelDir = Join-Path (Join-Path $modelRootDir $trainedDataset) $model
    
    if (-not (Test-Path $modelDir)) {
        Write-Warning "Skipping model '$model': Directory '$modelDir' does not exist."
        continue
    }
    
    $outDir = Join-Path $outParentDir $model

    $existingInferred = @()
    if (Test-Path $outDir) {
        $existingInferred = Get-ChildItem -Path $outDir -Filter "inferredChannel.mat" -Recurse -File -ErrorAction SilentlyContinue
    }
    $alreadyInferred = ($existingInferred.Count -gt 0)

    $inferScript = "inference_onnx_grid.py"
    if ($model -like "LS_Attention*" -or $model -like "LS_*") {
        $inferScript = "inference_onnx_lsSequence.py"
    }

    Write-Output "------------------------------------------------------------"
    Write-Output "Processing Inference Run $($i + 1)/$($models.Length):"
    Write-Output "  Model Name   : $model"
    Write-Output "  Infer Script : $inferScript"
    Write-Output "  Model Dir    : $modelDir"
    Write-Output "  Dataset Dir  : $datasetDir"
    Write-Output "  Output Dir   : $outDir"
    Write-Output "  Rerun Infer  : $($rerunInfer[$i])"

    if ($alreadyInferred -and -not $forceInfer) {
        Write-Output "  Status       : [SKIP] Found $($existingInferred.Count) existing inferredChannel.mat file(s)."
        Write-Output "                 Skipping Python inference (rerunInfer = 0)."
        Write-Output "------------------------------------------------------------`n"
        continue
    }

    if ($alreadyInferred -and $forceInfer) {
        Write-Output "  Status       : [RERUN] Results exist, but rerunInfer = 1. Re-running inference..."
    }
    else {
        Write-Output "  Status       : [NEW] Running Python ONNX inference..."
    }
    Write-Output "------------------------------------------------------------"
    
    conda run -n TF_GPU-py3_11 python $inferScript `
        --model-dir $modelDir `
        --dataset-dir $datasetDir `
        --out-dir $outDir `
        --num-samples $numSamples `
        --model-type $modelType `
        --clip-extrap $clipExtrap
        
    if ($LASTEXITCODE -eq 0) {
        Write-Output "Inference for '$model' completed successfully.`n"
    }
    else {
        Write-Warning "Inference for '$model' failed with exit code $LASTEXITCODE.`n"
    }
}

# 7. Stage 2: Run MATLAB batch evaluations
Write-Output "`n======================================================================"
Write-Output " Starting MATLAB Batch Performance Evaluations"
Write-Output "======================================================================`n"

$matlabExe = "matlab"
if (Test-Path "C:\Program Files\MATLAB\R2025a\bin\matlab.exe") {
    $matlabExe = "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
}

for ($i = 0; $i -lt $models.Length; $i++) {
    $model = $models[$i]
    $label = $labels[$i]
    $forceSyn = ($rerunSyn[$i] -eq 1) -or ($rerunSyn[$i] -eq $true)
    
    $evalFolder = Join-Path $outParentDir $model
    
    if (-not (Test-Path $evalFolder)) {
        Write-Warning "Skipping evaluation for model '$model': Path '$evalFolder' does not exist."
        continue
    }

    $synMatFile = Join-Path $evalFolder "synthesized_results.mat"
    $alreadySynthesized = Test-Path $synMatFile

    Write-Output "------------------------------------------------------------"
    Write-Output "Processing Evaluation ($($i + 1)/$($models.Length)):"
    Write-Output "  Model Folder : $evalFolder"
    Write-Output "  Model Label  : $label"
    Write-Output "  Rerun Flag   : $($rerunSyn[$i])"

    if ($alreadySynthesized -and -not $forceSyn) {
        Write-Output "  Status       : [SKIP] 'synthesized_results.mat' already exists."
        Write-Output "                 Skipping MATLAB evaluation (rerunSyn = 0)."
        Write-Output "------------------------------------------------------------`n"
        continue
    }

    if ($alreadySynthesized -and $forceSyn) {
        Write-Output "  Status       : [RERUN] 'synthesized_results.mat' exists, but rerunSyn = 1. Overwriting..."
    }
    else {
        Write-Output "  Status       : [NEW] Running MATLAB performance synthesis..."
    }
    Write-Output "------------------------------------------------------------"

    $escapedFolder = $evalFolder.Replace('\', '/')
    $matlabCmd = "syn_metrics_withBER('$escapedFolder', '$label')"

    & $matlabExe -batch $matlabCmd

    if ($LASTEXITCODE -eq 0) {
        Write-Output "Evaluation for '$model' completed successfully.`n"
    }
    else {
        Write-Warning "Evaluation for '$model' failed with exit code $LASTEXITCODE.`n"
    }
}

# 8. Stage 3: Multi-Model Consolidated Comparison
Write-Output "`n======================================================================"
Write-Output " Stage 3: Multi-Model Consolidated Comparison"
Write-Output "======================================================================`n"

$parentPath = $outParentDir
$validCompFolders = @()
$validLabels = @()

for ($i = 0; $i -lt $models.Length; $i++) {
    $model = $models[$i]
    $label = $labels[$i]
    $modelFolder = Join-Path $parentPath $model
    $synMat = Join-Path $modelFolder "synthesized_results.mat"

    if (Test-Path $synMat) {
        $validCompFolders += $modelFolder
        $validLabels += $label
    }
    else {
        Write-Warning "Excluding '$model' from comparison: '$synMat' not found."
    }
}

if ($validCompFolders.Length -eq 0) {
    Write-Error "No valid 'synthesized_results.mat' found among configured models! Skipping overall comparison."
}
else {
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

    Write-Output "Comparing $($validCompFolders.Length) model(s)..."
    Write-Output "Saving results to: $outputFolder`n"

    $matlabFoldersCell = "{" + (($validCompFolders | ForEach-Object { "'$( $_.Replace('\', '/') )'" }) -join ", ") + "}"
    $matlabLabelsCell = "{" + (($validLabels | ForEach-Object { "'$_'" }) -join ", ") + "}"
    $compareCmd = "syn_syn_compare_multiModels($matlabFoldersCell, $matlabLabelsCell, '$escapedOutFolder')"

    & $matlabExe -batch $compareCmd

    if ($LASTEXITCODE -eq 0) {
        Write-Output "Overall comparison completed successfully.`n"
    }
    else {
        Write-Warning "Overall comparison failed with exit code $LASTEXITCODE.`n"
    }
}

# 9. Create run summary note locally (done_infer.md)
$inferTriggerFile = Join-Path $outParentDir "done_infer.md"
$modelsStr = ($models | ForEach-Object { "- $_" }) -join "`n"
$labelsStr = ($labels | ForEach-Object { "- $_" }) -join "`n"
$trainedModelPath = Join-Path $modelRootDir $trainedDataset

$mdContent = @"
Inference & Evaluation completed on $(Get-Date -Format "MM/dd/yyyy HH:mm:ss")

Output Folder:
$outSaveFolderName

Source Model Path:
$trainedModelPath

Target Dataset Path:
$datasetDir

Models:
$modelsStr

Labels:
$labelsStr
"@

Set-Content -Path $inferTriggerFile -Value $mdContent -Encoding UTF8
Write-Output "Created summary file: $inferTriggerFile"
Write-Output "`n[DONE] Pipeline execution completed."
