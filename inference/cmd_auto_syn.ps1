# cmd_auto_syn.ps1
#
# OVERVIEW:
#   Automated end-to-end inference and evaluation pipeline across domain-shift scenarios:
#
#   1. Training Wait Loop (Git Polling - Optional):
#      Can poll Git looking for "done_train.md" inside the source model folder ($waitForTrain = $true),
#      or run immediately ($waitForTrain = $false).
#
#   2. Batch ONNX Inference:
#      Checks if inferredChannel.mat already exists for each model. Skips if already done ($rerunInfer = 0),
#      or re-runs Python ONNX inference ($rerunInfer = 1).
#
#   3. Local MATLAB Performance Evaluation:
#      Checks if synthesized_results.mat already exists for each model. Skips if already done ($rerunSyn = 0),
#      or re-runs syn_metrics_withBER in MATLAB ($rerunSyn = 1).
#
#   4. Multi-Model Consolidated Comparison:
#      Auto-detects the next incremental comparison folder (syn/syn_1, syn_2...),
#      and calls syn_syn_compare_multiModels in MATLAB to generate combined comparison curves and tables.
#
# RUN IN POWERSHELL:
#   .\inference\cmd_auto_syn.ps1
# --------------------------------------------------------------------------------------

# Automatically change directory to the folder containing this script
Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " Automated ONNX Inference & MATLAB Evaluation Pipeline (PowerShell)"
Write-Output "======================================================================`n"

# 1. Define root folder for trained models and the trained dataset parameter (Source)
$modelRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset"
$trainedDataset = "A100_2p18e9_600km_70deg_30kHz"

# 2. Define root folder for outputs and the folder to save results
$outRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\inference"
$outSaveFolderName = "A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz"

# 3. Define (Target) dataset directory (Common for all runs)
$datasetDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps"

# 4. List the models - the names of subfolders
$models = @(
    # "LI_cGAN",
    # "LI_cGAN_standardize",
    # "LI_DnCNN",
    # "LI_DnCNN_standardize",
    # "LI_DnCNN_Attention",
    # "LI_DnCNN_Attention_standardize",
    # "LI_DnCNN_AxialAttention",
    # "LI_DnCNN_AxialAttention_standardize",
    # "LI_DnCNN_CrossAttention",
    # "LI_DnCNN_CrossAttention_standardize",
    "LS_Attention",
    "LS_Attention_standardize"
    # "LS_Attention_AxialAttention",
    # "LS_Attention_AxialAttention_standardize",
    # "LS_Attention_cGAN",
    # "LS_Attention_cGAN_standardize",
    # "LS_Attention_DualDomain",
    # "LS_Attention_DualDomain_standardize",
    # "LS_Attention_ResidualRefine",
    # "LS_Attention_ResidualRefine_standardize",
    # "LS_Attention_UNetRefine",
    # "LS_Attention_UNetRefine_standardize",
    # "LS_DnCNN_Attention"
)

# Corresponding labels/legend names for evaluation
$labels = @(
    # "LI+cGAN Inferred",
    # "LI+cGAN Std Inferred",
    # "LI+DnCNN Inferred",
    # "LI+DnCNN Std Inferred",
    # "LI+DnCNN+Transformer Inferred",
    # "LI+DnCNN+Transformer Std Inferred",
    # "LI+DnCNN+AxialTransformer Inferred",
    # "LI+DnCNN+AxialTransformer Std Inferred",
    # "LI+DnCNN+CrossTransformer Inferred",
    # "LI+DnCNN+CrossTransformer Std Inferred",
    "LS+Transformer Inferred",
    "LS+Transformer Std Inferred"
    # "LS+Transformer+AxialTransformer Inferred",
    # "LS+Transformer+AxialTransformer Std Inferred",
    # "LS+Transformer+cGAN Inferred",
    # "LS+Transformer+cGAN Std Inferred",
    # "LS+Transformer+DualDomain Inferred",
    # "LS+Transformer+DualDomain Std Inferred",
    # "LS+Transformer+ResidualTransformer Inferred",
    # "LS+Transformer+ResidualTransformer Std Inferred",
    # "LS+Transformer+UNetTransformer Inferred",
    # "LS+Transformer+UNetTransformer Std Inferred",
    # "LS+Transformer+DnCNN Inferred"
)

# Rerun flags (must match $models length):
#   $rerunInfer: 0 = Skip Python inference if inferredChannel.mat exists, 1 = Force rerun inference
#   $rerunSyn  : 0 = Skip MATLAB evaluation if synthesized_results.mat exists, 1 = Force rerun MATLAB
$rerunInfer = @(
    0,
    0
)

$rerunSyn = @(
    0,
    0
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
# Set $waitForTrain = $false to run immediately without polling for done_train.md
$waitForTrain = $false
$trainTriggerFile = Join-Path (Join-Path $modelRootDir $trainedDataset) "done_train.md"
$checkIntervalSeconds = 1200  # Poll every 20 minutes

if ($waitForTrain) {
    Write-Output "Polling git pull every 20 minutes..."
    Write-Output "Looking for training completion flag: $trainTriggerFile`n"

    while ($true) {
        if (Test-Path $trainTriggerFile) {
            Write-Output "`n[TRIGGER DETECTED] Found done_train.md inside model folder!"
            break
        }

        Write-Output "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - done_train.md not found. Checking remote with git pull..."
        git pull

        if (Test-Path $trainTriggerFile) {
            Write-Output "`n[TRIGGER DETECTED] Found done_train.md inside model folder!"
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
    
    # Resolve full model directory path: $modelRootDir\$trainedDataset\$model
    $modelDir = Join-Path (Join-Path $modelRootDir $trainedDataset) $model
    
    # Check if the resolved model directory exists
    if (-not (Test-Path $modelDir)) {
        Write-Warning "Skipping model '$model': Directory '$modelDir' does not exist."
        continue
    }
    
    # Resolve output directory for this specific model
    $outDir = Join-Path $outParentDir $model

    # Check if inference results already exist (at least one inferredChannel.mat in SNR subfolders)
    $existingInferred = @()
    if (Test-Path $outDir) {
        $existingInferred = Get-ChildItem -Path $outDir -Filter "inferredChannel.mat" -Recurse -File -ErrorAction SilentlyContinue
    }
    $alreadyInferred = ($existingInferred.Count -gt 0)

    # Determine inference script dynamically (LS models use sequence inference)
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
    } else {
        Write-Output "  Status       : [NEW] Running Python ONNX inference..."
    }
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

# Automatically locate official MATLAB CLI launcher (prevents GUI detachment)
$matlabExe = "matlab"
if (Test-Path "C:\Program Files\MATLAB\R2025a\bin\matlab.exe") {
    $matlabExe = "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
}

for ($i = 0; $i -lt $models.Length; $i++) {
    $model = $models[$i]
    $label = $labels[$i]
    $forceSyn = ($rerunSyn[$i] -eq 1) -or ($rerunSyn[$i] -eq $true)
    
    # Construct absolute evaluation folder path
    $evalFolder = Join-Path $outParentDir $model
    
    if (-not (Test-Path $evalFolder)) {
        Write-Warning "Skipping evaluation for model '$model': Path '$evalFolder' does not exist."
        continue
    }

    $synMatFile = Join-Path $evalFolder "synthesized_results.mat"
    $alreadySynthesized = Test-Path $synMatFile

    Write-Output "------------------------------------------------------------"
    Write-Output "Evaluating Run $($i + 1)/$($models.Length):"
    Write-Output "  Model Folder : $model"
    Write-Output "  Plot Label   : $label"
    Write-Output "  Rerun Syn    : $($rerunSyn[$i])"

    if ($alreadySynthesized -and -not $forceSyn) {
        Write-Output "  Status       : [SKIP] 'synthesized_results.mat' already exists."
        Write-Output "                 Skipping MATLAB evaluation (rerunSyn = 0)."
        Write-Output "------------------------------------------------------------`n"
        continue
    }

    if ($alreadySynthesized -and $forceSyn) {
        Write-Output "  Status       : [RERUN] Results exist, but rerunSyn = 1. Overwriting..."
    } else {
        Write-Output "  Status       : [NEW] Running MATLAB syn_metrics_withBER..."
    }
    Write-Output "------------------------------------------------------------"
    
    $escapedFolder = $evalFolder.Replace('\', '/')
    & $matlabExe -batch "syn_metrics_withBER('$escapedFolder', '$label')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Output "Evaluation for '$model' completed successfully.`n"
    }
    else {
        Write-Warning "Evaluation for '$model' failed with exit code $LASTEXITCODE.`n"
    }
}

# 8. Stage 3: Multi-Model Consolidated Comparative Plots (syn_x)
Write-Output "`n======================================================================"
Write-Output " Running Multi-Model Consolidated Comparative Plots"
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
    } else {
        Write-Warning "Excluding '$model' from comparison: '$synMat' not found."
    }
}

if ($validCompFolders.Length -eq 0) {
    Write-Error "No valid 'synthesized_results.mat' found among configured models! Skipping overall comparison."
} else {
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

    # Build MATLAB cell arrays for folders and labels
    $matlabFoldersCell = "{" + (($validCompFolders | ForEach-Object { "'$( $_.Replace('\', '/') )'" }) -join ", ") + "}"
    $matlabLabelsCell = "{" + (($validLabels | ForEach-Object { "'$_'" }) -join ", ") + "}"
    $compareCmd = "syn_syn_compare_multiModels($matlabFoldersCell, $matlabLabelsCell, '$escapedOutFolder')"

    # Run overall comparison in MATLAB
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

$fileContent = @"
Inference & Evaluation completed on $(Get-Date)

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

Set-Content -Force -Path $inferTriggerFile -Value $fileContent
Write-Output "`nRun summary recorded in: $inferTriggerFile"

Write-Output "`n======================================================================"
Write-Output " All automated inference and evaluation runs completed successfully."
Write-Output "======================================================================"
