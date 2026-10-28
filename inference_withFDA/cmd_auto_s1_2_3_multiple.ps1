# cmd_auto_s1_2_3_multiple.ps1
#
# OVERVIEW:
#   Executes all FDA test-time approach pipelines sequentially:
#     1. Approach 2.5 : Iterative Self-Refinement (2-Pass FDA)
#     2. Approach 2.6 : Teacher-Student Hybrid Ensemble (Model 2 + Model 1)
#     3. Approach 3.2 : Model 2 Post-FDA Bias Correction
#     4. Approach 3.3 : Model 2 Pre-FDA Alignment + Post-FDA Correction
#
#   After all individual pipelines finish, it automatically generates a single
#   consolidated comparison plot (syn_*) comparing all 4 approaches side-by-side!
#
# RUN IN POWERSHELL:
#   .\inference_withFDA\cmd_auto_s1_2_3_multiple.ps1
# --------------------------------------------------------------------------------------

# Ensure script runs in its own directory regardless of where PowerShell was launched
Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " Starting Batch Execution of FDA Approaches (2.5 -> 2.6 -> 3.2 -> 3.3)"
Write-Output "======================================================================`n"

# 1. Run Approach 2.5
Write-Output "[1/4] Running Approach 2.5 (Iterative Self-Refinement)..."
& "$PSScriptRoot\cmd_auto_s1_2_3_2p5.ps1"
if ($LASTEXITCODE -ne 0) { Write-Warning "Approach 2.5 completed with exit code $LASTEXITCODE." }

# 2. Run Approach 2.6
Write-Output "`n[2/4] Running Approach 2.6 (Teacher-Student Hybrid Ensemble)..."
& "$PSScriptRoot\cmd_auto_s1_2_3_2p6.ps1"
if ($LASTEXITCODE -ne 0) { Write-Warning "Approach 2.6 completed with exit code $LASTEXITCODE." }

# 3. Run Approach 3.2
Write-Output "`n[3/4] Running Approach 3.2 (Model 2 Post-FDA Correction)..."
& "$PSScriptRoot\cmd_auto_s1_2_3_3p2.ps1"
if ($LASTEXITCODE -ne 0) { Write-Warning "Approach 3.2 completed with exit code $LASTEXITCODE." }

# 4. Run Approach 3.3
Write-Output "`n[4/4] Running Approach 3.3 (Model 2 Pre-FDA Alignment + Post-FDA Correction)..."
& "$PSScriptRoot\cmd_auto_s1_2_3_3p3.ps1"
if ($LASTEXITCODE -ne 0) { Write-Warning "Approach 3.3 completed with exit code $LASTEXITCODE." }

# --------------------------------------------------------------------
# Final Consolidated Comparison across All 4 Approaches
# --------------------------------------------------------------------
Write-Output "`n======================================================================"
Write-Output " Generating Master Comparison Plot Across All 4 Approaches"
Write-Output "======================================================================`n"

$outParentDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\inference_withFDA\A100_70deg__DUR300_30deg_2p18e9_600kmm_30kHz"

$compFolders = @(
    (Join-Path $outParentDir "LS_Attention_rms_app2p5_smooth13x3"),
    (Join-Path $outParentDir "LS_Attention_rms_app2p6_hybrid_13x3"),
    (Join-Path $outParentDir "LS_Attention_pseudo_app3p2_post_13x3"),
    (Join-Path $outParentDir "LS_Attention_pseudo_app3p3_dual_13x3")
)

$compLabels = @(
    "FDA 2.5 (Self-Refined)",
    "FDA 2.6 (Ensemble)",
    "FDA 3.2 (Pseudo Post)",
    "FDA 3.3 (Pseudo Dual)"
)

$validCompFolders = @()
$validLabels = @()
for ($i = 0; $i -lt $compFolders.Length; $i++) {
    $f = $compFolders[$i]
    if (Test-Path (Join-Path $f "synthesized_results.mat")) {
        $validCompFolders += $f
        $validLabels += $compLabels[$i]
    }
}

if ($validCompFolders.Length -gt 1) {
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
    $masterOutFolder = Join-Path $outParentDir "syn_$nextNum"
    $escapedMasterOutFolder = $masterOutFolder -replace '\\', '/'

    Write-Output "Comparing $($validCompFolders.Length) approaches together..."
    Write-Output "Saving master comparison to: $masterOutFolder`n"

    $matlabExe = "matlab"
    if (Test-Path "C:\Program Files\MATLAB\R2025a\bin\matlab.exe") {
        $matlabExe = "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
    }

    $matlabCompareCmd = "addpath('.'); syn_syn_compare_multiModels($matlabFoldersCell, $matlabLabelsCell, '$escapedMasterOutFolder'); exit;"
    & $matlabExe -batch $matlabCompareCmd
}

Write-Output "`n======================================================================"
Write-Output " All sequential runs and master comparison complete!"
Write-Output " Results saved under: $outParentDir"
Write-Output "======================================================================`n"
