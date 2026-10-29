# cmd_run_approachC.ps1
# Automates Approach C: Physics-Guided Attention Map CORAL UDA Training

Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " Approach C: Physics-Guided Attention CORAL (PowerShell Runner)"
Write-Output "======================================================================`n"

$sourceDir    = "..\..\generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz"
$targetDir    = "..\..\generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps"
$saveDir      = "..\results\approachC_physics_coral"
$snr          = 10
$epochs       = 100
$batchSize    = 16
$domainWeight = 1.0
$normMode     = "rms"      # "rms", "standardize", or "minmax"

$condaPy = "C:\Users\AT30890\AppData\Local\anaconda3\envs\TF_GPU-py3_11\python.exe"
if (-not (Test-Path $condaPy)) {
    $condaPy = "python"
}

Write-Output "Source Dir    : $sourceDir"
Write-Output "Target Dir    : $targetDir"
Write-Output "Save Dir      : $saveDir"
Write-Output "Channel SNR   : $snr dB"
Write-Output "Epochs        : $epochs"
Write-Output "Domain Weight : $domainWeight"
Write-Output "Normalization : $normMode"
Write-Output "Python Exe    : $condaPy`n"

& $condaPy train_physics_coral.py `
    --source-dir $sourceDir `
    --target-dir $targetDir `
    --save-dir $saveDir `
    --snr $snr `
    --epochs $epochs `
    --batch-size $batchSize `
    --domain-weight $domainWeight `
    --norm-mode $normMode

if ($LASTEXITCODE -eq 0) {
    Write-Output "`n[SUCCESS] Approach C training completed successfully! Results in $saveDir."
} else {
    Write-Warning "`n[ERROR] Approach C execution failed with exit code $LASTEXITCODE."
}

