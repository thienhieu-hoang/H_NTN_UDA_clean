# cmd_synSyn.ps1
#
# OVERVIEW:
#   Direct multi-model comparative analysis script for synthesized results.
#   Executes the MATLAB function "syn_syn_compare_multiModels" across all listed
#   model subfolders to compare MSE, NMSE, SSIM, and BER curves.
#
# PREREQUISITE:
#   This script is run AFTER performing the per-model MATLAB evaluation step
#   (e.g., via cmd_synMATLAB_metrics.ps1), which generates "synthesized_results.mat"
#   in each model folder.
#   No Git polling or .md trigger file check is needed.
#
# USAGE (PowerShell):
#   .\cmd_synSyn.ps1
# --------------------------------------------------------------------------------------

# Automatically change directory to the folder containing this script
Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " Multi-Model Consolidated Comparison (syn_syn_compare_multiModels)"
Write-Output "======================================================================`n"

# 1. Define fixed root path for inference runs
$inferenceRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\inference"

# 2. Define the chosen inferred runs directory
$inferredPath = "A100__DUR100_2p18e9_600km_30kHz"

# 3. List the model subfolders to compare
$models = @(
    # "LI_cGAN",
    # "LI_cGAN_standardize",
    # "LI_DnCNN",
    # "LI_DnCNN_standardize",
    # "LI_DnCNN_Attention",
    # "LI_DnCNN_Attention_standardize",
    # "LI_DnCNN_AxialAttention",
    # "LI_DnCNN_AxialAttention_standardize",
    "LI_DnCNN_CrossAttention",
    "LI_DnCNN_CrossAttention_standardize"
    # "LS_Attention",
    # "LS_Attention_standardize"
)

# 4. List the corresponding legend/plot label names for each model
$labels = @(
    # "LI+cGAN Inferred",
    # "LI+cGAN+Std Inferred",
    # "LI+DnCNN Inferred",
    # "LI+DnCNN+Std Inferred",
    # "LI+DnCNN+Transformer Inferred",
    # "LI+DnCNN+Transformer(Std) Inferred",
    # "LI+DnCNN+AxialTransformer Inferred",
    # "LI+DnCNN+AxialTransformer(Std) Inferred",
    "LI+DnCNN+CrossTransformer Inferred",
    "LI+DnCNN+CrossTransformer(Std) Inferred"
    # "LS+Transformer Inferred",
    # "LS+Transformer+Std Inferred"
)

# Verify list lengths match
if ($models.Length -ne $labels.Length) {
    Write-Error "Error: The number of models ($($models.Length)) does not match the number of labels ($($labels.Length))!"
    Exit
}

# 5. Locate official MATLAB CLI launcher
$matlabExe = "matlab"
if (Test-Path "C:\Program Files\MATLAB\R2025a\bin\matlab.exe") {
    $matlabExe = "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
}

# 6. Verify model paths and synthesized_results.mat availability
$parentPath = Join-Path $inferenceRootDir $inferredPath
Write-Output "Checking inputs in: $parentPath"
for ($i = 0; $i -lt $models.Length; $i++) {
    $modelFolder = Join-Path $parentPath $models[$i]
    $matFile = Join-Path $modelFolder "synthesized_results.mat"

    if (-not (Test-Path $modelFolder)) {
        Write-Warning "Model folder missing: $modelFolder"
    }
    elseif (-not (Test-Path $matFile)) {
        Write-Warning "Missing 'synthesized_results.mat' in '$($models[$i])'. Ensure the MATLAB sync step was run first."
    }
    else {
        Write-Output "  [OK] $($models[$i]) -> $($labels[$i])"
    }
}
Write-Output ""

# 7. Dynamic comparison output folder detection (syn_x)
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
Write-Output "Running Comparison Plots across all models..."
Write-Output "Destination : $outputFolder"
Write-Output "------------------------------------------------------------"

# 8. Build MATLAB cell arrays for folders and labels
$matlabFoldersCell = "{" + (($models | ForEach-Object { "'$( (Join-Path $parentPath $_).Replace('\', '/') )'" }) -join ", ") + "}"
$matlabLabelsCell = "{" + (($labels | ForEach-Object { "'$_'" }) -join ", ") + "}"
$compareCmd = "syn_syn_compare_multiModels($matlabFoldersCell, $matlabLabelsCell, '$escapedOutFolder')"

# 9. Run overall comparison in MATLAB batch mode
& $matlabExe -batch $compareCmd

if ($LASTEXITCODE -eq 0) {
    Write-Output "`n======================================================================"
    Write-Output " Multi-model comparison completed successfully."
    Write-Output " Comparative results and figures saved in: syn_$nextNum"
    Write-Output " Full Path: $outputFolder"
    Write-Output "======================================================================"
}
else {
    Write-Warning "`nMulti-model comparison failed with exit code $LASTEXITCODE."
}
