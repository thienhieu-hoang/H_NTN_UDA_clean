# cmd_syn_metrics_loop.ps1
#
# OVERVIEW:
#   This script automates performance evaluation for single-dataset models:
#
#   Loops through configured trained model subfolders (e.g. LI_DnCNN_AxialAttention)
#   and calls syn_results_withBER in MATLAB to synthesize evaluation metrics
#   (MSE, NMSE, SSIM, BER) across all SNR points.
#
# RUN IN POWERSHELL:
#   .\single_dataset\cmd_syn_metrics_loop.ps1
# --------------------------------------------------------------------------------------

# Automatically change directory to the folder containing this script
Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " MATLAB Batch Performance Evaluation Loop (single_dataset)"
Write-Output "======================================================================`n"

# 1. Define root folder for trained models and the trained dataset parameter
$modelRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset"
$trainedDataset = "A100_2p18e9_600km_70deg_30kHz"

# 2. List the model subfolders to evaluate
$models = @(
    "LI_cGAN",                              # 1
    "LI_cGAN_standardize",                  # 2
    "LI_DnCNN",                             # 3
    # "LI_DnCNN_standardize",                 # 4
    "LI_DnCNN_Attention",                   # 5
    "LI_DnCNN_Attention_standardize",       # 6
    "LI_DnCNN_AxialAttention",              # 7
    "LI_DnCNN_AxialAttention_standardize",  # 8
    "LI_DnCNN_CrossAttention",              # 9
    "LI_DnCNN_CrossAttention_standardize",  # 10
    "LS_Attention",                         # 11
    "LS_Attention_standardize",             # 12
    "LS_DnCNN_Attention"                    # 13
)

# Corresponding labels/legend names for evaluation
$labels = @(
    "LI+cGAN",                  # 1
    "LI+cGAN",                  # 2
    "LI+DnCNN",                 # 3
    # "LI+DnCNN",                 # 4
    "LI+DnCNN+Attention",       # 5
    "LI+DnCNN+Attention",       # 6
    "LI+DnCNN+AxialAttention",  # 7
    "LI+DnCNN+AxialAttention",  # 8
    "LI+DnCNN+CrossAttention",  # 9
    "LI+DnCNN+CrossAttention",  # 10
    "LS+Attention",             # 11
    "LS+Attention",             # 12
    "LS+DnCNN+Attention"        # 13
)

# Verify list lengths match
if ($models.Length -ne $labels.Length) {
    Write-Error "Error: The number of models ($($models.Length)) does not match the number of labels ($($labels.Length))!"
    Exit
}

# Automatically locate official MATLAB CLI launcher (prevents GUI detachment)
$matlabExe = "matlab"
if (Test-Path "C:\Program Files\MATLAB\R2025a\bin\matlab.exe") {
    $matlabExe = "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
}

# 3. Loop through each model and run the MATLAB evaluation function
for ($i = 0; $i -lt $models.Length; $i++) {
    $model = $models[$i]
    $label = $labels[$i]
    
    # Construct absolute evaluation folder path
    $evalFolder = Join-Path (Join-Path $modelRootDir $trainedDataset) $model
    
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

Write-Output "======================================================================"
Write-Output " All evaluation runs completed."
Write-Output "======================================================================"
