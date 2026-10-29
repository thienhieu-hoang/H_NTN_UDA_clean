# ==============================================================================
# Dataset Review & Physical Characterization Runner (Single & Batch SNR Loop)
# ==============================================================================

# Automatically resolve paths relative to the repository root
$scriptDir = $PSScriptRoot
$repoRoot  = (Resolve-Path "$scriptDir\..").Path
$pyScript  = Join-Path $scriptDir "review_dataset.py"

# ------------------------------------------------------------------------------
# 1. Configurable Dataset Root Folder to Review
# ------------------------------------------------------------------------------
# You can set either a relative path (to repo root) OR a full absolute path:

# Option A (Math Pseudo):
# $datasetInput = "generatedChan_mathPseudo\A100_DUR300_LS"

# Option B (Source A100):
# $datasetInput = "generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz"

# Option C (Target DUR300 - with -10dB, 0dB, 5dB subfolders):
$datasetInput = "generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps"

# Smart Path Resolver: Handles both absolute ("C:\...") and relative paths
if ([System.IO.Path]::IsPathRooted($datasetInput)) {
    $datasetRoot = $datasetInput
} else {
    $datasetRoot = Join-Path $repoRoot $datasetInput
}

# ------------------------------------------------------------------------------
# 2. SNR Configuration (Loop or Single Folder)
# ------------------------------------------------------------------------------
# Set to an array of SNRs to loop automatically: @(-10, -5, 0, 5, 10, 15)
# Set to $null if $datasetRoot is already pointing directly to a single folder (e.g. ...\SNR_5dB or ...\5dB)
$snrList    = @(-10, -5, 0, 5, 10, 15)

# Other Settings:
$scsKhz     = 30.0       # 5G NR Subcarrier Spacing (30 kHz)
$numSamples = $null      # e.g., 200 for fast preview, or $null for all samples
$condaEnv   = "TF_GPU-py3_11"

# ------------------------------------------------------------------------------
# Helper Function to locate SNR folder name variants adaptively
# Supports: -10dB, 5dB, +5dB, SNR_-10dB, SNR_5dB, LS_-10dB, etc.
# ------------------------------------------------------------------------------
function Find-SnrFolder ($parentDir, $snr) {
    if (-not (Test-Path $parentDir)) {
        return $null
    }

    # 1. Direct candidate paths
    $candidates = @(
        "${snr}dB",
        "SNR_${snr}dB",
        "SNR_${snr}",
        "LS_${snr}dB",
        "LS_${snr}",
        "+${snr}dB",
        "SNR_+${snr}dB",
        "${snr}"
    )
    foreach ($cand in $candidates) {
        $testP = Join-Path $parentDir $cand
        if (Test-Path $testP) {
            return $testP
        }
    }

    # 2. Adaptive regex scan across all subdirectories
    $subs = Get-ChildItem -Path $parentDir -Directory
    foreach ($s in $subs) {
        $name = $s.Name.Trim()
        # Matches patterns like: -10dB, 5dB, +5dB, SNR_-10dB, SNR_5dB, LS_0dB, _5dB
        if ($name -match '(?i)(?:SNR_|LS_|_)?([+-]?\d+)(?:dB)?$') {
            try {
                if ([int]$Matches[1] -eq $snr) {
                    return $s.FullName
                }
            } catch {}
        }
    }
    return $null
}

# ------------------------------------------------------------------------------
# 3. Execution (Loop over SNRs or single folder)
# ------------------------------------------------------------------------------
$targetFolders = @()

if ($snrList -ne $null -and $snrList.Count -gt 0) {
    Write-Host "Auto-discovering SNR folders under: $datasetRoot" -ForegroundColor Cyan
    foreach ($snr in $snrList) {
        $snrFolder = Find-SnrFolder $datasetRoot $snr
        if ($snrFolder) {
            $targetFolders += $snrFolder
        } else {
            Write-Warning "Could not find subfolder for SNR = $snr dB under $datasetRoot"
        }
    }
} else {
    # Single folder mode
    $targetFolders += $datasetRoot
}

if ($targetFolders.Count -eq 0) {
    Write-Error "No valid dataset folders found to review. Please check `$datasetRoot and `$snrList."
    exit 1
}

Write-Host "Found $($targetFolders.Count) folder(s) to review:`n" -ForegroundColor Green

$idx = 1
foreach ($folder in $targetFolders) {
    Write-Host "[$idx/$($targetFolders.Count)] ========================================================" -ForegroundColor Yellow
    Write-Host "  Reviewing: $folder" -ForegroundColor Yellow
    Write-Host "=====================================================================" -ForegroundColor Yellow

    $argList = @(
        "$pyScript",
        "--dataset-dir", "$folder",
        "--scs-khz", "$scsKhz"
    )
    if ($numSamples -ne $null) {
        $argList += @("--num-samples", "$numSamples")
    }

    if ($condaEnv) {
        conda run -n $condaEnv python @argList
    } else {
        python @argList
    }

    Write-Host ""
    $idx++
}

Write-Host "All reviews completed successfully!" -ForegroundColor Green
