# run_approach_A.ps1
# Automates Approach A: Physical Pseudo-Dataset Generation via Delay Dilation and Doppler Modulation

Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " Approach A: Physical Pseudo-Dataset Generation (PowerShell Runner)"
Write-Output "======================================================================`n"

$sourceDir  = "..\..\generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz"
$targetDir  = "..\..\generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps"
$outDir     = "..\results\approachA_pseudo_dataset"
$scaleMode  = "rms"       # "rms" or "standardize"
$numSamples = "None"      # Limit number of samples or "None"

$condaPy = "C:\Users\AT30890\AppData\Local\anaconda3\envs\TF_GPU-py3_11\python.exe"
if (-not (Test-Path $condaPy)) {
    $condaPy = "python"
}

Write-Output "Source Dir : $sourceDir"
Write-Output "Target Dir : $targetDir"
Write-Output "Output Dir : $outDir"
Write-Output "Scale Mode : $scaleMode"
Write-Output "Python Exe : $condaPy`n"

& $condaPy generate_pseudo_dataset.py `
    --source-dir $sourceDir `
    --target-dir $targetDir `
    --out-dir $outDir `
    --scale-mode $scaleMode `
    --snr-list -10 -5 0 5 10 15

if ($LASTEXITCODE -eq 0) {
    Write-Output "`n[SUCCESS] Approach A pseudo dataset generated successfully at $outDir!"
} else {
    Write-Warning "`n[ERROR] Approach A generation failed with exit code $LASTEXITCODE."
}

