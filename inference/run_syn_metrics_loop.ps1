# run_batch_metrics_loop.ps1
# Place this inside the inference/ folder
# Run in PowerShell: .\run_batch_metrics_loop.ps1

# Automatically change directory to the folder containing this script
Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " MATLAB Batch Performance Evaluation Loop (PowerShell)"
Write-Output "======================================================================`n"

# 1. Define fixed root path for inference runs
$inferenceRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\inference"

# 2. Define the chosen inferred runs directory
$inferredPath = "DUR100__A100_2p18e9_600km_30kHz"

# 3. List the model subfolders to evaluate
$models = @(
    "LI_DnCNN_AxialAttention",
    "LI_DnCNN_AxialAttention_standardize",
    "LI_DnCNN_CrossAttention",
    "LI_DnCNN_CrossAttention_standardize"
)

# 4. List the corresponding legend/plot label names for each model
$labels = @(
    "LI+DnCNN+AxialAttention Inferred",
    "LI+DnCNN+AxialAttention+Standardize Inferred",
    "LI+DnCNN+CrossAttention Inferred",
    "LI+DnCNN+CrossAttention+Standardize Inferred"
)

# Verify list lengths match
if ($models.Length -ne $labels.Length) {
    Write-Error "Error: The number of models ($($models.Length)) does not match the number of labels ($($labels.Length))!"
    Exit
}

# 5. Loop through each model and run the MATLAB evaluation function
for ($i = 0; $i -lt $models.Length; $i++) {
    $model = $models[$i]
    $label = $labels[$i]
    
    # Construct absolute evaluation folder path
    $evalFolder = Join-Path (Join-Path $inferenceRootDir $inferredPath) $model
    
    # Double check if folder exists before invoking MATLAB (optional safety check)
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
    
    # Invoke MATLAB in headless batch mode, calling the syn_metrics_withBER function
    # Note: we double escape the backslashes inside single quotes for MATLAB strings
    $escapedFolder = $evalFolder -replace '\\', '\\'
    
    matlab -batch "syn_metrics_withBER('$escapedFolder', '$label')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Output "Evaluation for '$model' completed successfully.`n"
    } else {
        Write-Warning "Evaluation for '$model' failed with exit code $LASTEXITCODE.`n"
    }
}

Write-Output "======================================================================"
Write-Output " All evaluation runs completed."
Write-Output "======================================================================"
