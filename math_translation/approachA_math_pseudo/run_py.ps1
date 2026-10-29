# ==============================================================================
# Approach A: Physical Pseudo-Dataset Generation Runner
# ==============================================================================

# Automatically resolve paths relative to the repository root
$scriptDir = $PSScriptRoot
$repoRoot  = (Resolve-Path "$scriptDir\..\..").Path
$pyScript  = Join-Path $scriptDir "generate_pseudo_dataset.py"

# 1. Configurable Paths (relative to repo root or absolute)
$sourceDir = Join-Path $repoRoot "generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz"
$targetDir = Join-Path $repoRoot "generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps"
$outDir    = Join-Path $repoRoot "generatedChan_mathPseudo\A100_DUR300_LS"

# 2. Configurable Generation Settings
$snrList   = @(-10, -5, 0, 5, 10, 15)
$scaleMode = "rms"        # options: "rms", "standardize", "none"
$numSamples = $null       # e.g., 200 for quick testing, or $null for all samples

# 3. Optional Manual Parameter Overrides (Leave $null to auto-estimate from pilots)
$betaTau   = $null        # e.g., 3.0
$deltaNu   = $null        # e.g., 32.5

# 4. Conda Environment (Default project env with compatible NumPy 1.x & SciPy)
$condaEnv  = "TF_GPU-py3_11"  # Set to $null if you already activated the env in your terminal

# ==============================================================================
# Build Arguments & Execute Python
# ==============================================================================
$argList = @(
    "$pyScript",
    "--source-dir", "$sourceDir",
    "--target-dir", "$targetDir",
    "--out-dir", "$outDir",
    "--scale-mode", "$scaleMode",
    "--snr-list"
) + $snrList

if ($numSamples -ne $null) {
    $argList += @("--num-samples", "$numSamples")
}
if ($betaTau -ne $null) {
    $argList += @("--beta-tau", "$betaTau")
}
if ($deltaNu -ne $null) {
    $argList += @("--delta-nu", "$deltaNu")
}

if ($condaEnv) {
    Write-Host "Running in conda env '$condaEnv': python $($argList -join ' ')" -ForegroundColor Cyan
    conda run -n $condaEnv python @argList
} else {
    Write-Host "Running: python $($argList -join ' ')" -ForegroundColor Cyan
    python @argList
}
