# cmd_auto_syn.ps1
#
# OVERVIEW:
#   Automated end-to-end evaluation & multi-model comparison pipeline for CORAL UDA models:
#
#   1. Training Wait Loop (Git Polling - Optional):
#      Can poll Git looking for "done_train.md" inside the dataset folder ($waitForTrain = $true),
#      or run immediately ($waitForTrain = $false).
#
#   2. Local Model & Layer Evaluation (syn_metrics_withBER):
#      Evaluates each model and its extracted layers/blocks.
#      - If extract layers are specified, evaluates those exact layer subfolders.
#      - If extract layers are empty or undefined, automatically discovers all subfolders matching 'block*' or 'layer*'.
#      - Checks if 'synthesized_results.mat' already exists:
#          * rerun = 0 -> Skips MATLAB evaluation (fast reuse).
#          * rerun = 1 -> Re-runs syn_metrics_withBER and overwrites.
#
#   3. Consolidated Comparative Analysis (syn_syn_compare_multiModels):
#      Gathers all valid evaluated layer folders across configured models:
#      - If 1 model: saves layer comparison directly inside model directory (<dataset>\<model>\syn_1, syn_2...).
#      - If multiple models: saves multi-model comparison in dataset directory (<dataset>\syn\syn_1, syn_2...).
#      Runs syn_syn_compare_multiModels in MATLAB to generate publication-quality comparison curves.
#
# RUN IN POWERSHELL:
#   .\CORAL\cmd_auto_syn.ps1
# --------------------------------------------------------------------------------------
# Note on available values for $models (in A100_2p18e9_600km_70deg_30kHz):
#   "LI_cGAN",
#   "LI_cGAN_standardize",
#   "LI_DnCNN",
#   "LI_DnCNN_standardize",
#   "LI_DnCNN_Attention",
#   "LI_DnCNN_Attention_standardize",
#   "LI_DnCNN_AxialAttention",
#   "LI_DnCNN_AxialAttention_standardize",
#   "LI_DnCNN_CrossAttention",
#   "LI_DnCNN_CrossAttention_standardize",
#   "LS_Attention",
#   "LS_Attention_standardize",
#   "LS_Attention_AxialAttention",
#   "LS_Attention_AxialAttention_standardize",
#   "LS_Attention_cGAN",
#   "LS_Attention_cGAN_standardize",
#   "LS_Attention_DualDomain",
#   "LS_Attention_DualDomain_standardize",
#   "LS_Attention_ResidualRefine",
#   "LS_Attention_ResidualRefine_standardize",
#   "LS_Attention_UNetRefine",
#   "LS_Attention_UNetRefine_standardize",
#
# and corresponding labels:
#   "LI+cGAN",
#   "LI+cGAN std",
#   "LI+DnCNN",
#   "LI+DnCNN std",
#   "LI+DnCNN+Transformer",
#   "LI+DnCNN+Transformer std",
#   "LI+DnCNN+AxialTransformer",
#   "LI+DnCNN+AxialTransformer std",
#   "LI+DnCNN+CrossTransformer",
#   "LI+DnCNN+CrossTransformer std",
#   "LS+Transformer",
#   "LS+Transformer std",
#   "LS+Transformer+AxialTransformer",
#   "LS+Transformer+AxialTransformer std",
#   "LS+Transformer+cGAN",
#   "LS+Transformer+cGAN std",
#   "LS+Transformer+DualDomain",
#   "LS+Transformer+DualDomain std",
#   "LS+Transformer+ResidualTransformer",
#   "LS+Transformer+ResidualTransformer std",
#   "LS+Transformer+UNetTransformer",
#   "LS+Transformer+UNetTransformer std",
# --------------------------------------------------------------------------------------

# Automatically change directory to the folder containing this script
Set-Location $PSScriptRoot

Write-Output "======================================================================"
Write-Output " CORAL UDA Automated Performance Evaluation & Multi-Model Plotting Loop"
Write-Output "======================================================================`n"

# 1. Define root folder for CORAL experiments and the target dataset scenario
$coralRootDir = "C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\CORAL"
$dataset = "A100__DUR100_2p18e9_600km_30kHz"

# 2. List the models to evaluate
$models = @(
    "LI_DnCNN"
)

# Corresponding base legend/plot label names for each model
$labels = @(
    "LI+DnCNN CORAL"
)

# 3. Define layer/block subfolders to evaluate for each model:
#    - Specify subfolder names: e.g. @("block3", "block2_block3") or @("layer1_layer2")
#    - Set as empty @() or $null to AUTOMATICALLY discover all matching 'block*' and 'layer*' subfolders!
$extractLayers = @(
    @()                       # Empty = auto-discover block2_block3, block3
)

# 4. Rerun flags for each model (must match $models length):
#    0 = Skip MATLAB evaluation if synthesized_results.mat already exists (reuse existing)
#    1 = Force rerun MATLAB evaluation and overwrite existing results
$rerun = @(
    0
)

# Auto-fill / normalize $labels if missing
if ($labels -eq $null -or $labels.Length -eq 0) {
    $labels = $models
}

# Auto-wrap / normalize $extractLayers to avoid PowerShell's empty-array flattening gotcha
if ($extractLayers -eq $null -or $extractLayers.Length -eq 0) {
    $normLayers = [object[]]::new($models.Length)
    for ($idx = 0; $idx -lt $models.Length; $idx++) {
        $normLayers[$idx] = @()
    }
    $extractLayers = $normLayers
}
elseif ($models.Length -eq 1 -and $extractLayers.Length -eq 1 -and ($extractLayers[0] -is [string])) {
    $normLayers = [object[]]::new(1)
    $normLayers[0] = @($extractLayers[0])
    $extractLayers = $normLayers
}

# Auto-fill $rerun if missing
if ($rerun -eq $null -or $rerun.Length -eq 0) {
    $rerun = @(0) * $models.Length
}

# Verify list lengths match
if ($models.Length -ne $labels.Length -or $models.Length -ne $extractLayers.Length -or $models.Length -ne $rerun.Length) {
    Write-Error "Error: The lengths of `$models ($($models.Length)), `$labels ($($labels.Length)), `$extractLayers ($($extractLayers.Length)), and `$rerun ($($rerun.Length)) must match!"
    Exit 1
}

# 5. Remote training completion polling configuration
# Set $waitForTrain = $false to run evaluation immediately without waiting for done_train.md
$waitForTrain = $false
$datasetDir = Join-Path $coralRootDir $dataset
$trainTriggerFile = Join-Path $datasetDir "done_train.md"
$checkIntervalSeconds = 1200  # Poll every 20 minutes

if ($waitForTrain) {
    Write-Output "Polling git pull every 20 minutes..."
    Write-Output "Looking for training completion flag: $trainTriggerFile`n"

    while ($true) {
        if (Test-Path $trainTriggerFile) {
            Write-Output "`n[TRIGGER DETECTED] Found done_train.md inside dataset folder!"
            break
        }

        Write-Output "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - done_train.md not found. Checking remote with git pull..."
        git pull

        if (Test-Path $trainTriggerFile) {
            Write-Output "`n[TRIGGER DETECTED] Found done_train.md inside dataset folder!"
            break
        }

        Write-Output "No done_train.md found. Waiting 20 minutes..."
        Start-Sleep -Seconds $checkIntervalSeconds
    }
}

# Automatically locate official MATLAB CLI launcher (prevents GUI detachment)
$matlabExe = "matlab"
if (Test-Path "C:\Program Files\MATLAB\R2025a\bin\matlab.exe") {
    $matlabExe = "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
}

# Helper to format layer/block folder names cleanly for plot legends without underscores:
# e.g., "block2_block3" -> "block 2, 3", "layer1_layer2" -> "layer 1, 2", "block3" -> "block 3"
function Format-LayerLegend ($layerName) {
    if ([string]::IsNullOrWhiteSpace($layerName)) { return "" }

    if ($layerName -match '^(block|layer)') {
        $prefix = $Matches[1]
        $numbers = [regex]::Matches($layerName, '\d+') | ForEach-Object { $_.Value }
        if ($numbers.Count -gt 0) {
            return "$prefix $($numbers -join ', ')"
        }
    }
    # Fallback for other names: replace underscores with spaces
    return ($layerName -replace '_', ' ')
}

# 6. Local Batch Performance Evaluation Loop
Write-Output "`n======================================================================"
Write-Output " Starting Local Model & Layer Performance Evaluations (syn_metrics_withBER)"
Write-Output "======================================================================`n"

$allTargetFolders = @()
$allTargetLabels = @()

for ($i = 0; $i -lt $models.Length; $i++) {
    $model = $models[$i]
    $baseLabel = $labels[$i]
    $reqLayers = $extractLayers[$i]
    $forceRerun = ($rerun[$i] -eq 1) -or ($rerun[$i] -eq $true)

    $modelDir = Join-Path $datasetDir $model
    if (-not (Test-Path $modelDir)) {
        Write-Warning "Skipping model '$model': Path '$modelDir' does not exist."
        continue
    }

    # Resolve layer subfolders (explicitly specified vs auto-discovered)
    $targetLayers = @()
    if ($reqLayers -ne $null -and $reqLayers.Count -gt 0) {
        foreach ($ly in $reqLayers) {
            $candidatePath = Join-Path $modelDir $ly
            if (Test-Path $candidatePath) {
                $targetLayers += $ly
            }
            else {
                # Intelligent alias fallback: check if swapping 'layer' <-> 'block' resolves the subfolder
                $swappedLy = $null
                if ($ly -match "layer") {
                    $swappedLy = $ly -replace "layer", "block"
                }
                elseif ($ly -match "block") {
                    $swappedLy = $ly -replace "block", "layer"
                }

                if ($swappedLy -and (Test-Path (Join-Path $modelDir $swappedLy))) {
                    Write-Output "  [Auto-Mapped] Requested '$ly' -> '$swappedLy' for model '$model'."
                    $targetLayers += $swappedLy
                }
                else {
                    Write-Warning "Requested layer '$ly' not found in '$modelDir'."
                }
            }
        }
    }
    else {
        # Auto-discover all subfolders matching 'block*' or 'layer*' that are not comparison folders ('syn*')
        $discovered = Get-ChildItem -Path $modelDir -Directory | Where-Object {
            ($_.Name -like "block*" -or $_.Name -like "layer*") -and ($_.Name -notlike "syn*")
        }
        if ($discovered.Count -gt 0) {
            $targetLayers = $discovered | Select-Object -ExpandProperty Name
        }
        else {
            # Fallback: check if model directory directly contains SNR subfolders
            $snrCheck = Get-ChildItem -Path $modelDir -Directory | Where-Object {
                $_.Name -like "LS_*" -or $_.Name -like "LI_*"
            }
            if ($snrCheck.Count -gt 0) {
                $targetLayers = @("") # Direct model folder
            }
        }
    }

    if ($targetLayers.Count -eq 0) {
        Write-Warning "No valid layer/block subfolders found for model '$model'. Skipping."
        continue
    }

    Write-Output "------------------------------------------------------------"
    Write-Output "Processing Model [$($i + 1)/$($models.Length)]: $model"
    Write-Output "  Base Label   : $baseLabel"
    Write-Output "  Target Layers: $($targetLayers -join ', ')"
    Write-Output "  Rerun Flag   : $($rerun[$i])"
    Write-Output "------------------------------------------------------------"

    foreach ($layer in $targetLayers) {
        $layerDir = if ($layer -eq "") { $modelDir } else { Join-Path $modelDir $layer }
        $cleanLayer = Format-LayerLegend $layer
        $layerLabel = if ($cleanLayer -eq "") { $baseLabel } else { "$baseLabel ($cleanLayer)" }

        # Check if synthesized_results.mat already exists
        $synMatFile = Join-Path $layerDir "synthesized_results.mat"
        $alreadySynthesized = Test-Path $synMatFile

        if ($alreadySynthesized -and -not $forceRerun) {
            Write-Output "  [SKIP] '$layer': synthesized_results.mat already exists."
            Write-Output "         Path: $layerDir (rerun = 0)`n"
        }
        else {
            if ($alreadySynthesized -and $forceRerun) {
                Write-Output "  [RERUN] '$layer': Results exist, but rerun = 1. Overwriting..."
            }
            else {
                Write-Output "  [NEW] '$layer': Running MATLAB syn_metrics_withBER..."
            }
            Write-Output "         Folder: $layerDir"

            $escapedFolder = $layerDir.Replace('\', '/')
            & $matlabExe -batch "syn_metrics_withBER('$escapedFolder', '$layerLabel')"

            if ($LASTEXITCODE -eq 0) {
                Write-Output "  Evaluation for '$layer' completed successfully.`n"
            }
            else {
                Write-Warning "  Evaluation for '$layer' failed with exit code $LASTEXITCODE.`n"
            }
        }

        # If results exist (either from this run or previously), add to comparison queue
        if (Test-Path $synMatFile) {
            $allTargetFolders += $layerDir
            $allTargetLabels += $layerLabel
        }
    }
}

# 7. Multi-Model Consolidated Comparative Plots
Write-Output "======================================================================"
Write-Output " Generating Multi-Model Comparative Plots (syn_syn_compare_multiModels)"
Write-Output "======================================================================`n"

if ($allTargetFolders.Count -eq 0) {
    Write-Error "No valid synthesized results found to compare! Skipping overall comparison."
    Exit 1
}

# Resolve comparison destination directory:
# - If 1 model is configured: save inside the local model folder (<dataset>\<model>\syn_<n>)
# - If multiple models are configured: save inside the dataset-level syn folder (<dataset>\syn\syn_<n>)
$isSingleModel = ($models.Length -eq 1)

if ($isSingleModel) {
    $synParentDir = Join-Path $datasetDir $models[0]
    Write-Output "Single model configured ('$($models[0])')."
    Write-Output "Saving comparative results inside model directory: $synParentDir"
}
else {
    $synParentDir = Join-Path $datasetDir "syn"
    if (-not (Test-Path $synParentDir)) {
        New-Item -ItemType Directory -Force -Path $synParentDir | Out-Null
    }
    Write-Output "Multiple models configured ($($models.Length))."
    Write-Output "Saving consolidated results in dataset comparison directory: $synParentDir"
}

$maxNum = 0
$subDirs = Get-ChildItem -Path $synParentDir -Directory -Filter "syn*"
foreach ($dir in $subDirs) {
    if ($dir.Name -match '^syn_?(\d+)$') {
        $num = [int]$Matches[1]
        if ($num -gt $maxNum) {
            $maxNum = $num
        }
    }
}
$nextNum = $maxNum + 1
$compareOutFolder = Join-Path $synParentDir "syn_$nextNum"
$escapedCompareOut = $compareOutFolder.Replace('\', '/')

Write-Output "Comparing $($allTargetFolders.Count) layer/model target(s)..."
Write-Output "Output Directory: $compareOutFolder`n"

# Build MATLAB cell arrays
$matlabFoldersCell = "{" + (($allTargetFolders | ForEach-Object { "'$( $_.Replace('\', '/') )'" }) -join ", ") + "}"
$matlabLabelsCell = "{" + (($allTargetLabels  | ForEach-Object { "'$_'" }) -join ", ") + "}"
$compareCmd = "syn_syn_compare_multiModels($matlabFoldersCell, $matlabLabelsCell, '$escapedCompareOut')"

& $matlabExe -batch $compareCmd

if ($LASTEXITCODE -eq 0) {
    Write-Output "`nOverall comparative plots completed successfully.`n"
}
else {
    Write-Warning "`nOverall comparative plots failed with exit code $LASTEXITCODE.`n"
}

Write-Output "======================================================================"
Write-Output " Done! All local evaluations and multi-model comparisons finished."
Write-Output " Consolidated output saved to: $compareOutFolder"
Write-Output "======================================================================"
