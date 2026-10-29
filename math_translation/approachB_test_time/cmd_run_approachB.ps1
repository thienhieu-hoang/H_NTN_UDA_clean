# cmd_run_approachB.ps1
# Automates Approach B: Test-Time Domain Adaptation via Physical Coordinate Warping (Zero Retraining)

Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " Approach B: Test-Time Warping Adaptation (PowerShell Runner)"
Write-Output "======================================================================`n"

$modelDir         = "..\..\single_dataset\A100_2p18e9_600km_70deg_30kHz\LS_Attention_rms"
$targetDatasetDir = "..\..\generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps"
$sourceRefDir     = "..\..\generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz"
$outDir           = "..\results\approachB_test_time"
$pilotAnchorBlend = 1.0     # 1.0 = exact pilot substitution; 0.0 = no anchoring
$numSamples       = "None"  # "None" for all samples, or integer (e.g. 100)

$condaPy = "C:\Users\AT30890\AppData\Local\anaconda3\envs\TF_GPU-py3_11\python.exe"
if (-not (Test-Path $condaPy)) {
    $condaPy = "python"
}

Write-Output "Model Dir       : $modelDir"
Write-Output "Target Dataset  : $targetDatasetDir"
Write-Output "Output Dir      : $outDir"
Write-Output "Pilot Anchoring : $pilotAnchorBlend"
Write-Output "Python Exe      : $condaPy`n"

& $condaPy test_time_warping_infer.py `
    --model-dir $modelDir `
    --target-dataset-dir $targetDatasetDir `
    --source-ref-dir $sourceRefDir `
    --out-dir $outDir `
    --pilot-anchor-blend $pilotAnchorBlend

if ($LASTEXITCODE -eq 0) {
    Write-Output "`n[SUCCESS] Approach B evaluation completed successfully! Results in $outDir."
} else {
    Write-Warning "`n[ERROR] Approach B execution failed with exit code $LASTEXITCODE."
}

