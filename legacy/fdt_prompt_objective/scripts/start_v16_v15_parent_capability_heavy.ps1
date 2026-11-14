param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Checkpoint = Join-Path $Root "runs\fdt_v3_capability_completion_v15_r2\latest.pt"
$CheckpointHash = "E8D5C00CAEC9E13D4A5BB78B5DDA15A94B7AA8497D37CF0C3586F148FBAF47CE"
$Dataset = Join-Path $Root "prepared_data\v3_capability_completion_v16_master_6750_ctx512"
$ManifestHash = "CA5B487EFA172927D27975902A8FC2E521CC138DE032DA745C73F19EEB2CE68F"
$Selection = Join-Path $Root "artifacts\fdt_v16_v15_parent_retention_mix_selection_20260813.json"
$SelectionHash = "F194D930381C6931E648E11832CD4800F354ED406A7C9F358DAFD1482859333C"
$OutputDir = Join-Path $Root "runs\fdt_v3_capability_completion_v16_v15_parent_capability_heavy_20260813"

foreach ($Path in @($Root, $Checkpoint, $Dataset, $Selection, $OutputDir)) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "V16 training permits only C: paths: $Path"
    }
}
if ((Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash -ne $CheckpointHash) {
    throw "Preserved V15 checkpoint hash mismatch"
}
if ((Get-FileHash -LiteralPath (Join-Path $Dataset "manifest.json") -Algorithm SHA256).Hash -ne $ManifestHash) {
    throw "V16 dataset manifest hash mismatch"
}
if ((Get-FileHash -LiteralPath $Selection -Algorithm SHA256).Hash -ne $SelectionHash) {
    throw "V16 selection artifact hash mismatch"
}
$SelectionData = Get-Content -LiteralPath $Selection -Raw | ConvertFrom-Json
if ($SelectionData.status -ne "SELECTED" -or $SelectionData.selected_profile -ne "capability_heavy" -or [double]$SelectionData.selected_prompt_lm_weight -ne 0.1) {
    throw "Pinned V16 selection is not capability_heavy with prompt weight 0.1"
}
if (Test-Path -LiteralPath $OutputDir) {
    if (Get-ChildItem -LiteralPath $OutputDir -Force) { throw "Dedicated V16 output directory is not empty" }
} else {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $OutputDir "train.$Stamp.stdout.log"
$Stderr = Join-Path $OutputDir "train.$Stamp.stderr.log"
$Arguments = @(
    "-u",
    "scripts\train_capability_completion_v16.py",
    "--checkpoint", $Checkpoint,
    "--expected-checkpoint-sha256", $CheckpointHash,
    "--dataset-dir", $Dataset,
    "--expected-manifest-sha256", $ManifestHash,
    "--selection-artifact", $Selection,
    "--expected-selection-sha256", $SelectionHash,
    "--output-dir", $OutputDir,
    "--batch-size", "8",
    "--lr", "0.0000001",
    "--smoke-step", "100",
    "--max-broad-relative-regression", "0.003",
    "--max-prefix-repetition-increase", "0.01",
    "--max-loop-free-decrease", "0.05",
    "--max-capability-loss-regression", "0.0",
    "--max-dead-anchor-ratio", "0.01",
    "--seed", "20260816",
    "--device", "cuda"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $OutputDir "train.pid") -Encoding ascii
@{
    stage = "v16_v15_parent_capability_heavy"
    pid = $Process.Id
    started_at = (Get-Date).ToString("o")
    stdout = $Stdout
    stderr = $Stderr
    checkpoint = $Checkpoint
    checkpoint_sha256 = $CheckpointHash
    dataset = $Dataset
    manifest_sha256 = $ManifestHash
    selection = $Selection
    selection_sha256 = $SelectionHash
    selected_profile = "capability_heavy"
    prompt_lm_weight = 0.1
    output_dir = $OutputDir
    training = $true
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDir "active_logs.json") -Encoding utf8

Write-Output "Started V16 V15-parent capability-heavy training PID $($Process.Id)"
