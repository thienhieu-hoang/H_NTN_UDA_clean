# cmd_syn_syn_results.ps1
#
# OVERVIEW:
#   This script automates multi-model comparative performance plotting by running
#   syn_syn_results_.m in MATLAB for single_dataset benchmark results.
#
#   It gathers pre-synthesized results (synthesized_results.mat) across selected
#   model subfolders (e.g. LI_cGAN, LI_DnCNN, LS_Attention), builds the corresponding
#   MATLAB cell arrays, auto-increments the output folder (syn/syn_1, syn/syn_2, ...),
#   and generates unified comparative PDF figures (MSE, NMSE, SSIM, BER).
#
# USAGE (PowerShell):
#   .\single_dataset\cmd_syn_syn_results.ps1
# --------------------------------------------------------------------------------------

# Automatically change directory to the folder containing this script
Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " MATLAB Comparative Multi-Model Plotting Loop (syn_syn_results_)"
Write-Output "======================================================================`n"

# 1. Define root folder for trained models and the trained dataset scenario
$modelRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset"
$trainedDataset = "A100_2p18e9_600km_70deg_30kHz"
# $trainedDataset = "DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps"

# 2. List the model subfolders to compare
$models = @(
    # "LI_cGAN",                              # 1
    # "LI_cGAN_standardize",                  # 2
    # "LI_DnCNN",                             # 3
    # "LI_DnCNN_standardize"                 # 4
    # "LI_DnCNN_Attention",                   # 5
    # "LI_DnCNN_Attention_standardize"       # 6
    # "LI_DnCNN_AxialAttention",              # 7
    # "LI_DnCNN_AxialAttention_standardize"  # 8
    # "LI_DnCNN_CrossAttention",              # 9
    # "LI_DnCNN_CrossAttention_standardize"  # 10
    "LS_Attention",                         # 11
    "LS_Attention_standardize"             # 12
    # "LS_DnCNN_Attention"                    # 13
)

# Corresponding labels/legend names for evaluation plots
$labels = @(
    # "LI+cGAN",                              # 1
    # "LI+cGAN (Std)",                        # 2
    # "LI+DnCNN",                             # 3
    # "LI+DnCNN (Std)"                       # 4
    # "LI+DnCNN+Transformer",                   # 5
    # "LI+DnCNN+Transformer (Std)"             # 6
    # "LI+DnCNN+AxialTransformer",              # 7
    # "LI+DnCNN+AxialTransformer (Std)"        # 8
    # "LI+DnCNN+CrossTransformer",              # 9
    # "LI+DnCNN+CrossTransformer (Std)"        # 10
    "LS+Transformer",                         # 11
    "LS+Transformer (Std)"                   # 12
    # "LS+DnCNN+Transformer"                    # 13
)

# Optional custom output folder (leave empty "" to automatically use incremental <dataset>/syn/syn_x)
$customOutputFolder = ""

# Verify list lengths match
if ($models.Length -ne $labels.Length) {
    Write-Error "Error: The number of models ($($models.Length)) does not match the number of labels ($($labels.Length))!"
    Exit 1
}

# 3. Locate official MATLAB executable
$matlabExe = "matlab"
if (Test-Path "C:\Program Files\MATLAB\R2025a\bin\matlab.exe") {
    $matlabExe = "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
}

# 4. Resolve synthesize subfolder paths and check existence of synthesized_results.mat
$validCompFolders = @()
$validLabels = @()

$datasetBase = Join-Path $modelRootDir $trainedDataset

Write-Output "Target Dataset Scenario: $trainedDataset"
Write-Output "Resolving synthesized results for $($models.Length) configured model(s)...`n"

for ($i = 0; $i -lt $models.Length; $i++) {
    $model = $models[$i]
    $label = $labels[$i]
    $modelDir = Join-Path $datasetBase $model

    if (-not (Test-Path $modelDir)) {
        Write-Warning "Skipping '$model': Directory does not exist -> $modelDir"
        continue
    }

    # Identify synthesis subfolder
    $synSubfolder = $null
    $candidates = @(
        (Join-Path $modelDir "LI_synthesize"),
        (Join-Path $modelDir "LS_synthesize"),
        $modelDir
    )

    foreach ($cand in $candidates) {
        $matFile = Join-Path $cand "synthesized_results.mat"
        if (Test-Path $matFile) {
            $synSubfolder = $cand
            break
        }
    }

    if ($synSubfolder -ne $null) {
        $validCompFolders += $synSubfolder
        $validLabels += $label
        Write-Output "  [OK] Found results for: $model -> $synSubfolder"
    }
    else {
        Write-Warning "  [Not Found] Missing 'synthesized_results.mat' in '$model'. Run cmd_syn_metrics_loop.ps1 first."
    }
}

if ($validCompFolders.Length -eq 0) {
    Write-Error "`nError: No valid 'synthesized_results.mat' files found among configured models."
    Exit 1
}

Write-Output "`nTotal valid models to compare: $($validCompFolders.Length)/$($models.Length)"

# 5. Resolve Output Directory (Incremental syn_x or custom folder)
if ($customOutputFolder -ne "" -and $customOutputFolder -ne $null) {
    $compareOutFolder = $customOutputFolder
}
else {
    $parentPath = Join-Path $datasetBase "syn"
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
}

if (-not (Test-Path $compareOutFolder)) {
    New-Item -ItemType Directory -Force -Path $compareOutFolder | Out-Null
}

$escapedCompareOut = $compareOutFolder.Replace('\', '/')

# 6. Format MATLAB cell arrays and command
$matlabFoldersCell = "{" + (($validCompFolders | ForEach-Object { "'$( $_.Replace('\', '/') )'" }) -join ", ") + "}"
$matlabLabelsCell = "{" + (($validLabels      | ForEach-Object { "'$_'" }) -join ", ") + "}"
$compareCmd = "syn_syn_results_($matlabFoldersCell, $matlabLabelsCell, '$escapedCompareOut')"

Write-Output "`n======================================================================"
Write-Output " Launching MATLAB Comparison Function (syn_syn_results_)..."
Write-Output " Output Folder: $compareOutFolder"
Write-Output "======================================================================`n"

& $matlabExe -batch $compareCmd

if ($LASTEXITCODE -eq 0) {
    Write-Output "`n======================================================================"
    Write-Output " [SUCCESS] Overall comparative plots generated successfully!"
    Write-Output " Results saved to: $compareOutFolder"
    Write-Output "======================================================================"
}
else {
    Write-Warning "`n[ERROR] syn_syn_results_ failed with exit code $LASTEXITCODE."
}
