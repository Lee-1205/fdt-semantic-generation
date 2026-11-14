param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Checkpoint = Join-Path $Root "runs\fdt_v3_capability_completion_v15_r2\latest.pt"
$CheckpointHash = "E8D5C00CAEC9E13D4A5BB78B5DDA15A94B7AA8497D37CF0C3586F148FBAF47CE"
$Dataset = Join-Path $Root "prepared_data\v3_capability_completion_v16_master_6750_ctx512"
$ManifestHash = "CA5B487EFA172927D27975902A8FC2E521CC138DE032DA745C73F19EEB2CE68F"
$RunDir = Join-Path $Root "runs\fdt_v16_v15_parent_retention_mix_selector_20260813"
$Output = Join-Path $Root "artifacts\fdt_v16_v15_parent_retention_mix_selection_20260813.json"

foreach ($Path in @($Root, $Checkpoint, $Dataset, $RunDir, $Output)) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "V16 selector permits only C: paths: $Path"
    }
}
if ((Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash -ne $CheckpointHash) {
    throw "Preserved V15 checkpoint hash mismatch"
}
if ((Get-FileHash -LiteralPath (Join-Path $Dataset "manifest.json") -Algorithm SHA256).Hash -ne $ManifestHash) {
    throw "V16 dataset manifest hash mismatch"
}
if ((Get-Content -LiteralPath (Join-Path $Dataset "manifest.json") -Raw | ConvertFrom-Json).post_build_exact_overlap -ne 0) {
    throw "V16 dataset overlap audit is not zero"
}
if (Test-Path -LiteralPath $Output) { throw "V16 V15-parent selection artifact already exists" }
if (Test-Path -LiteralPath "$Output.partial") { throw "V16 V15-parent partial artifact already exists" }
if (Test-Path -LiteralPath $RunDir) {
    if (Get-ChildItem -LiteralPath $RunDir -Force) { throw "V16 selector run directory is not empty" }
} else {
    New-Item -ItemType Directory -Path $RunDir | Out-Null
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $RunDir "selector.$Stamp.stdout.log"
$Stderr = Join-Path $RunDir "selector.$Stamp.stderr.log"
$Arguments = @(
    "-u",
    "scripts\select_capability_completion_v16.py",
    "--checkpoint", $Checkpoint,
    "--expected-checkpoint-sha256", $CheckpointHash,
    "--dataset-dir", $Dataset,
    "--expected-manifest-sha256", $ManifestHash,
    "--output", $Output,
    "--smoke-steps", "75",
    "--batch-size", "8",
    "--capability-rows", "160",
    "--lr", "0.0000001",
    "--max-broad-relative-regression", "0.0015",
    "--max-prefix-repetition-increase", "0.01",
    "--max-loop-free-decrease", "0.10",
    "--seed", "20260816",
    "--device", "cuda"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $RunDir "selector.pid") -Encoding ascii
@{
    stage = "v16_v15_parent_retention_mix_selector"
    pid = $Process.Id
    started_at = (Get-Date).ToString("o")
    stdout = $Stdout
    stderr = $Stderr
    checkpoint = $Checkpoint
    checkpoint_sha256 = $CheckpointHash
    dataset = $Dataset
    manifest_sha256 = $ManifestHash
    output = $Output
    partial_output = "$Output.partial"
    training = $true
    final_training_launched = $false
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunDir "active_logs.json") -Encoding utf8

Write-Output "Started V16 V15-parent retention-mix selector PID $($Process.Id)"
