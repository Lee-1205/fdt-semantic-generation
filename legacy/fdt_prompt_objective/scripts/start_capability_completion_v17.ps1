param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Checkpoint = Join-Path $Root "runs\fdt_v3_capability_completion_v16_v15_parent_capability_heavy_20260813\latest.pt"
$CheckpointHash = "7883BA4EC7AEE545B2CA9CBB395F2A357166811C7070C688942210F4B8864FF2"
$Dataset = Join-Path $Root "prepared_data\v3_capability_completion_v17_balanced_5k_ctx512"
$ManifestHash = "9069A02C2A1101010302F5227FB58C20DD8FD215C23856790BCB3AAB5F1D663A"
$Trainer = Join-Path $Root "scripts\train_capability_completion_v17.py"
$TrainerHash = "62086E7D482FC619E13F3FA66077885C3BEAE9397471A3F521CCC5F967221FCA"
$OutputDir = Join-Path $Root "runs\fdt_v3_capability_completion_v17_balanced_20260814"

foreach ($Path in @($Root, $Checkpoint, $Dataset, $Trainer, $OutputDir)) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "V17 permits only C: paths: $Path"
    }
}
if ((Get-PSDrive C).Free -lt 82GB) { throw "V17 launch requires at least 82 GiB free on C:" }
if ((Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash -ne $CheckpointHash) { throw "V16 parent hash mismatch" }
if ((Get-FileHash -LiteralPath (Join-Path $Dataset "manifest.json") -Algorithm SHA256).Hash -ne $ManifestHash) { throw "V17 manifest hash mismatch" }
if ((Get-FileHash -LiteralPath $Trainer -Algorithm SHA256).Hash -ne $TrainerHash) { throw "V17 trainer hash mismatch" }
if (Test-Path -LiteralPath $OutputDir) {
    if (Get-ChildItem -LiteralPath $OutputDir -Force) { throw "Dedicated V17 output is not empty" }
} else {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $OutputDir "train.$Stamp.stdout.log"
$Stderr = Join-Path $OutputDir "train.$Stamp.stderr.log"
$Arguments = @(
    "-u", "scripts\train_capability_completion_v17.py",
    "--checkpoint", $Checkpoint,
    "--expected-checkpoint-sha256", $CheckpointHash,
    "--dataset-dir", $Dataset,
    "--expected-manifest-sha256", $ManifestHash,
    "--output-dir", $OutputDir,
    "--batch-size", "10",
    "--lr", "0.00000005",
    "--weight-decay", "0.01",
    "--prompt-lm-weight", "0.2",
    "--smoke-step", "100",
    "--max-broad-relative-regression", "0.003",
    "--max-prefix-repetition-increase", "0.01",
    "--max-loop-free-decrease", "0.05",
    "--max-capability-loss-regression", "0.0",
    "--max-dead-anchor-ratio", "0.01",
    "--seed", "20260817",
    "--device", "cuda"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $OutputDir "train.pid") -Encoding ascii
@{
    stage = "v17_balanced_category_normalized"
    pid = $Process.Id
    started_at = (Get-Date).ToString("o")
    stdout = $Stdout
    stderr = $Stderr
    checkpoint = $Checkpoint
    checkpoint_sha256 = $CheckpointHash
    dataset = $Dataset
    manifest_sha256 = $ManifestHash
    trainer_sha256 = $TrainerHash
    output_dir = $OutputDir
    training = $true
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDir "active_logs.json") -Encoding utf8

Write-Output "Started balanced V17 training PID $($Process.Id)"
