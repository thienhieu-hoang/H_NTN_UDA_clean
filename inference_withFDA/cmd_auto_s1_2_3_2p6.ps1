# cmd_auto_s1_2_3_2p6.ps1
#
# OVERVIEW:
#   End-to-End Automated Pipeline for Approach 2.6 / 3.4 (Teacher-Student Hybrid Ensemble):
#   - Step 1: Model 2 (trained on pseudo-dataset) generates initial target macro Doppler/delay prior.
#   - Step 2: Model 1 (trained on source ground truth) receives target input disguised with Model 2 prior,
#             and performs sharp multipath and phase denoising.
#   - Step 3: Post-FDA restoration with physical target input + Pilot Consistency Projection (PCP).
#
# RUN IN POWERSHELL:
#   .\inference_withFDA\cmd_auto_s1_2_3_2p6.ps1
# --------------------------------------------------------------------------------------

# Automatically change directory to the folder containing this script
Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " Approach 2.6: Teacher-Student Hybrid Ensemble (Model 2 + Model 1)"
Write-Output "======================================================================`n"

# 1. Define root folder for trained models
$modelRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset"

# Model 1 (Phase / High-frequency Expert - Clean Source trained)
$trainedDataset1 = "A100_2p18e9_600km_70deg_30kHz"
$model1Name = "LS_Attention_rms"

# Model 2 (Target Macro Doppler/Delay Prior Expert - Pseudo dataset trained)
$trainedDataset2 = "pseudo_A100Perfect_DUR300LSAttention_rms_13x3"
$model2Name = "LS_Attention_rmsFDA_rmsReTrain"

# 2. Define root folder for outputs and the common scenario folder
$outRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\inference_withFDA"
$outSaveFolderName = "A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz"

# 3. Define (Target) dataset directory
$datasetDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps"

# 4. Define Source Reference Dataset Directory
$sourceRefPath = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz"

# 5. FDA Parameters for Approach 2.6
$approach = "2.6"
$windowType = "tukey_outward"
$winH = 13
$winW = 3
$taperH = 2
$taperW = 1
$pcpAlpha = 0.5

# 6. Output Subfolder and Plot Label
$models = @(
    $model1Name
)

$outSubfolders = @(
    "LS_Attention_rms_app2p6_hybrid_13x3"
)

$labels = @(
    "LS+Transformer FDA 2.6 (Ensemble)"
)

# Rerun flags:
$rerunInfer = @(
    1
)

$rerunSyn = @(
    1
)

# Optional inference hyperparameters
$numSamples = $null       # Number of samples (e.g. 64 for quick test, or $null for all)
$modelType = "auto"   # "auto", "LS", "LI"
$clipExtrap = "auto"   # "auto", "true", "false"

# Resolve full output parent directory
$outParentDir = Join-Path $outRootDir $outSaveFolderName

# Resolve full model directories
$model1Dir = Join-Path (Join-Path $modelRootDir $trainedDataset1) $model1Name
$model2Dir = Join-Path (Join-Path $modelRootDir $trainedDataset2) $model2Name

Write-Output "Pipeline Configuration:"
Write-Output "  FDA Approach         : Approach $approach (Hybrid Ensemble)"
Write-Output "  Model 1 (Phase)      : $model1Dir"
Write-Output "  Model 2 (Macro Prior): $model2Dir"
Write-Output "  Target Dataset Path  : $datasetDir"
Write-Output "  Output Destination   : $outParentDir"
Write-Output "  FDA Window Type      : $windowType (${winH}x${winW}, Taper: ${taperH}x${taperW})"
Write-Output "  Pilot Consistency    : Alpha = $pcpAlpha"
Write-Output "  Output Subfolder     : $($outSubfolders -join ', ')"
Write-Output "======================================================================`n"

# --------------------------------------------------------------------
# Stage 1: Batch ONNX Inference with Approach 2.6
# --------------------------------------------------------------------
Write-Output "`n======================================================================"
Write-Output " Stage 1: Batch ONNX Inference with Approach $approach"
Write-Output "======================================================================`n"

for ($i = 0; $i -lt $models.Length; $i++) {
    $model = $models[$i]
    $outSubfolder = if ($outSubfolders -and $outSubfolders[$i]) { $outSubfolders[$i] } else { $model }
    $forceInfer = ($rerunInfer[$i] -eq 1) -or ($rerunInfer[$i] -eq $true)
    
    if (-not (Test-Path $model1Dir)) {
        Write-Warning "Model 1 directory '$model1Dir' does not exist."
        continue
    }
    if (-not (Test-Path $model2Dir)) {
        Write-Warning "Model 2 directory '$model2Dir' does not exist."
        continue
    }
    
    $outDir = Join-Path $outParentDir $outSubfolder

    $existingInferred = @()
    if (Test-Path $outDir) {
        $existingInferred = Get-ChildItem -Path $outDir -Filter "inferredChannel.mat" -Recurse -File -ErrorAction SilentlyContinue
    }
    $alreadyInferred = ($existingInferred.Count -gt 0)

    Write-Output "------------------------------------------------------------"
    Write-Output "Processing Ensemble:"
    Write-Output "  Model 1 Dir : $model1Dir"
    Write-Output "  Model 2 Dir : $model2Dir"
    Write-Output "  Output Dir  : $outDir"

    if ($alreadyInferred -and -not $forceInfer) {
        Write-Output "  Status      : [SKIP] Found $($existingInferred.Count) existing inferredChannel.mat file(s)."
        Write-Output "------------------------------------------------------------`n"
        continue
    }

    Write-Output "  Status      : Running inference_onnx_with_fda.py (Approach $approach)..."
    Write-Output "------------------------------------------------------------"

    $extraArgs = @()
    if ($null -ne $numSamples) {
        $extraArgs += @("--num-samples", "$numSamples")
    }

    conda run -n TF_GPU-py3_11 python inference_onnx_with_fda.py `
        --model-dir $model1Dir `
        --model2-dir $model2Dir `
        --dataset-dir $datasetDir `
        --out-dir $outDir `
        --source-ref-path $sourceRefPath `
        --approach $approach `
        --window-type $windowType `
        --win-h $winH `
        --win-w $winW `
        --taper-h $taperH `
        --taper-w $taperW `
        --pcp-alpha $pcpAlpha `
        --clip-extrap $clipExtrap `
        @extraArgs
}

# Automatically locate official MATLAB CLI launcher (prevents GUI detachment & streams progress)
$matlabExe = "matlab"
if (Test-Path "C:\Program Files\MATLAB\R2025a\bin\matlab.exe") {
    $matlabExe = "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
}

# --------------------------------------------------------------------
# Stage 2: MATLAB Performance Evaluation (syn_metrics_withBER)
# --------------------------------------------------------------------
Write-Output "`n======================================================================"
Write-Output " Stage 2: MATLAB Performance Evaluation (syn_metrics_withBER)"
Write-Output "======================================================================`n"

for ($i = 0; $i -lt $models.Length; $i++) {
    $outSubfolder = if ($outSubfolders -and $outSubfolders[$i]) { $outSubfolders[$i] } else { $models[$i] }
    $label = $labels[$i]
    $forceSyn = ($rerunSyn[$i] -eq 1) -or ($rerunSyn[$i] -eq $true)
    $targetDir = Join-Path $outParentDir $outSubfolder

    if (-not (Test-Path $targetDir)) {
        Write-Warning "Skipping evaluation for '$outSubfolder': Directory '$targetDir' does not exist."
        continue
    }

    $synMatPath = Join-Path $targetDir "synthesized_results.mat"
    $alreadySynthesized = Test-Path $synMatPath

    Write-Output "------------------------------------------------------------"
    Write-Output "Evaluating Subfolder: $outSubfolder"
    Write-Output "  Label     : $label"
    Write-Output "  Directory : $targetDir"

    if ($alreadySynthesized -and -not $forceSyn) {
        Write-Output "  Status    : [SKIP] synthesized_results.mat already exists."
        Write-Output "------------------------------------------------------------`n"
        continue
    }

    Write-Output "  Status    : Executing MATLAB syn_metrics_withBER..."
    Write-Output "------------------------------------------------------------"

    $escapedPath = $targetDir -replace '\\', '/'
    $matlabCmd = "addpath('.'); syn_metrics_withBER('$escapedPath', '$label'); exit;"
    & $matlabExe -batch $matlabCmd
}

# --------------------------------------------------------------------
# Stage 3: Multi-Model Consolidated Comparison
# --------------------------------------------------------------------
Write-Output "`n======================================================================"
Write-Output " Stage 3: Multi-Model Consolidated Comparison"
Write-Output "======================================================================`n"

$validCompFolders = @()
$validLabels = @()
for ($i = 0; $i -lt $models.Length; $i++) {
    $sub = if ($outSubfolders -and $outSubfolders[$i]) { $outSubfolders[$i] } else { $models[$i] }
    $evalFolder = Join-Path $outParentDir $sub
    $synMatFile = Join-Path $evalFolder "synthesized_results.mat"
    if (Test-Path $synMatFile) {
        $validCompFolders += $evalFolder
        $validLabels += $labels[$i]
    }
}

if ($validCompFolders.Length -eq 0) {
    Write-Warning "No valid synthesized_results.mat found to compare. Skipping overall comparison."
}
else {
    $maxNum = 0
    if (Test-Path $outParentDir) {
        $subDirs = Get-ChildItem -Path $outParentDir -Directory -Filter "syn_*"
        foreach ($dir in $subDirs) {
            if ($dir.Name -match '^syn_(\d+)$') {
                $num = [int]$Matches[1]
                if ($num -gt $maxNum) { $maxNum = $num }
            }
        }
    }
    $nextNum = $maxNum + 1
    $outputFolder = Join-Path $outParentDir "syn_$nextNum"
    $escapedOutFolder = $outputFolder -replace '\\', '/'

    Write-Output "Comparing $($validCompFolders.Length) model(s)..."
    Write-Output "Saving results to: $outputFolder`n"

    $matlabFoldersCell = "{" + (($validCompFolders | ForEach-Object { "'$( $_ -replace '\\', '/' )'" }) -join ", ") + "}"
    $matlabLabelsCell = "{" + (($validLabels | ForEach-Object { "'$_'" }) -join ", ") + "}"

    $matlabCompareCmd = "addpath('.'); syn_syn_compare_multiModels($matlabFoldersCell, $matlabLabelsCell, '$escapedOutFolder'); exit;"
    Write-Output "Executing MATLAB comparison for: $outParentDir"
    & $matlabExe -batch $matlabCompareCmd
}

Write-Output "`n======================================================================"
Write-Output " Pipeline Complete! Results saved under: $outParentDir"
Write-Output "======================================================================`n"
