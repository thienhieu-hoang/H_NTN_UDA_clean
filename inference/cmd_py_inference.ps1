# run_batch_inference_loop.ps1
# Place this at the same level as inference_onnx_grid.py (inside the inference/ directory)
# Run in PowerShell: .\run_batch_inference_loop.ps1

# Automatically change directory to the folder containing this script
Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " ONNX Multi-Model Batch Inference Loop (PowerShell)"
Write-Output "======================================================================`n"

# 1. Define (Target) dataset directory (Common for all runs)
$datasetDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps"

# 2. Define root folder for trained models and the trained dataset parameter
$modelRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset"
$trainedDataset = "DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps"

# 3. List the models you want to run inference on
# The script will search for directories named "${model}_${trainedDataset}" under $modelRootDir,
# or fall back to "${model}" directly if the suffix is not used.
$models = @(
    "LI_DnCNN_AxialAttention",
    "LI_DnCNN_AxialAttention_standardize",
    "LI_DnCNN_CrossAttention",
    "LI_DnCNN_CrossAttention_standardize"
)

# 4. Define root folder for outputs and the folder to save results
$outRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\inference"
$outSaveFolderName = "DUR100__A100_2p18e9_600km_30kHz"

# Other common parameters
$numSamples = "None"          # Limit number of samples (or "None" to process all)
$modelType = "auto"           # "LS", "LI", "PRAC", or "auto"
$clipExtrap = "auto"        # "auto", "true", or "false"

# Create output parent directory if it does not exist
$outParentDir = Join-Path $outRootDir $outSaveFolderName
if (-not (Test-Path $outParentDir)) {
    New-Item -ItemType Directory -Force -Path $outParentDir | Out-Null
}

# 5. Loop through each model and run inference
for ($i = 0; $i -lt $models.Length; $i++) {
    $model = $models[$i]
    
    # Resolve full model directory path: $modelRootDir\$trainedDataset\$model
    $modelDir = Join-Path (Join-Path $modelRootDir $trainedDataset) $model
    
    # Check if the resolved model directory exists
    if (-not (Test-Path $modelDir)) {
        Write-Warning "Skipping model '$model': Directory '$modelDir' does not exist."
        continue
    }
    
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

Write-Output "======================================================================"
Write-Output " All batch inference runs completed."
Write-Output "======================================================================"
