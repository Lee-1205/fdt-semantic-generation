param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Checkpoint = Join-Path $Root "runs\fdt_v3_capability_completion_v17_balanced_20260814\latest.pt"
$CheckpointHash = "A7C418B6014412E39D6C9C659C7614A0677D7D062CE1A6A7239A7DD8CF0A8D92"
$Dataset = Join-Path $Root "prepared_data\v3_capability_completion_v18_calibrated_5k_ctx512"
$ManifestHash = "8AA96A89EDAECCB752B5C40546E97719CF2012DF8E02778FCD0A4F2925EA39F0"
$TrainShardHash = "E3179AB9BB8C576A444508B5A09FB3E33E039368FDCDF78EAB9820992E9A7EBE"
$ValidationShardHash = "E044E8B8D2382825994A1115FC95A5553ADF556DE503CB4DBA83BFC6CDC1FFE6"
$Trainer = Join-Path $Root "scripts\train_capability_completion_v18.py"
$TrainerHash = "BE2AA78263F1B8D93210AA20C86867209E67F4398693DBE1755464FF1749E778"
$OutputDir = Join-Path $Root "runs\fdt_v3_capability_completion_v18_calibrated_retention_20260814"

foreach ($Path in @($Root, $Checkpoint, $Dataset, $Trainer, $OutputDir)) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "V18 permits only C: paths: $Path"
    }
}
if ((Get-PSDrive C).Free -lt 82GB) { throw "V18 launch requires at least 82 GiB free on C:" }
if ((Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash -ne $CheckpointHash) { throw "V17 parent hash mismatch" }
if ((Get-FileHash -LiteralPath (Join-Path $Dataset "manifest.json") -Algorithm SHA256).Hash -ne $ManifestHash) { throw "V18 manifest hash mismatch" }
if ((Get-FileHash -LiteralPath (Join-Path $Dataset "shards\train\shard_000000.pt") -Algorithm SHA256).Hash -ne $TrainShardHash) { throw "V18 train shard hash mismatch" }
if ((Get-FileHash -LiteralPath (Join-Path $Dataset "shards\validation\shard_000000.pt") -Algorithm SHA256).Hash -ne $ValidationShardHash) { throw "V18 validation shard hash mismatch" }
if ((Get-FileHash -LiteralPath $Trainer -Algorithm SHA256).Hash -ne $TrainerHash) { throw "V18 trainer hash mismatch" }

$ActiveTrainer = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'train_capability_completion|train_scaled'
}
if ($ActiveTrainer) { throw "another FDT trainer is already active: $($ActiveTrainer.ProcessId -join ',')" }
if (Test-Path -LiteralPath $OutputDir) {
    if (Get-ChildItem -LiteralPath $OutputDir -Force) { throw "Dedicated V18 output is not empty" }
} else {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $OutputDir "train.$Stamp.stdout.log"
$Stderr = Join-Path $OutputDir "train.$Stamp.stderr.log"
$Arguments = @(
    "-u", "scripts\train_capability_completion_v18.py",
    "--checkpoint", $Checkpoint,
    "--expected-checkpoint-sha256", $CheckpointHash,
    "--dataset-dir", $Dataset,
    "--expected-manifest-sha256", $ManifestHash,
    "--output-dir", $OutputDir,
    "--batch-size", "10",
    "--lr", "0.00000005",
    "--weight-decay", "0.01",
    "--prompt-lm-weight", "0.25",
    "--eos-loss-weight", "1.25",
    "--smoke-step", "100",
    "--max-broad-relative-regression", "0.003",
    "--max-prefix-repetition-increase", "0.01",
    "--max-loop-free-decrease", "0.05",
    "--max-capability-loss-regression", "0.0",
    "--max-dead-anchor-ratio", "0.01",
    "--seed", "20260818",
    "--device", "cuda"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $OutputDir "train.pid") -Encoding ascii
@{
    stage = "v18_calibrated_retention"
    pid = $Process.Id
    started_at = (Get-Date).ToString("o")
    stdout = $Stdout
    stderr = $Stderr
    checkpoint = $Checkpoint
    checkpoint_sha256 = $CheckpointHash
    dataset = $Dataset
    manifest_sha256 = $ManifestHash
    train_shard_sha256 = $TrainShardHash
    validation_shard_sha256 = $ValidationShardHash
    trainer_sha256 = $TrainerHash
    output_dir = $OutputDir
    training = $true
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDir "active_logs.json") -Encoding utf8

Write-Output "Started calibrated-retention V18 training PID $($Process.Id)"
