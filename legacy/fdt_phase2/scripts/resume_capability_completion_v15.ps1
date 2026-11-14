param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$RunDir = Join-Path $Root "runs\fdt_v3_capability_completion_v15_r2"
$Checkpoint = Join-Path $RunDir "latest.pt"
$CheckpointHash = "7C4FACE3B27902307FCBF4BC5C0EFBC6BA319394C62BDFE02F1E258F51EA0E70"
$DataDir = Join-Path $Root "prepared_data\v3_capability_completion_v15_5k_ctx512"
$ManifestHash = "97FC204DA290100E0CEF1E7B542276CECB30E8894492774B9FDE10BCEED62A85"
$StopFile = Join-Path $RunDir "STOP_REQUESTED"

foreach ($Path in @($Root, $RunDir, $Checkpoint, $DataDir)) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "V15 permits only C: paths: $Path"
    }
}
if (-not (Test-Path -LiteralPath $Checkpoint)) { throw "Missing paused V15 checkpoint" }
if (Test-Path -LiteralPath (Join-Path $RunDir "latest.pt.tmp")) { throw "Incomplete atomic checkpoint exists" }
if ((Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash -ne $CheckpointHash) {
    throw "Paused V15 checkpoint hash mismatch"
}
if ((Get-FileHash -LiteralPath (Join-Path $DataDir "manifest.json") -Algorithm SHA256).Hash -ne $ManifestHash) {
    throw "V15 manifest hash mismatch"
}
$OldPid = if (Test-Path -LiteralPath (Join-Path $RunDir "train.pid")) {
    [int](Get-Content -LiteralPath (Join-Path $RunDir "train.pid"))
} else { 0 }
if ($OldPid -and (Get-Process -Id $OldPid -ErrorAction SilentlyContinue)) {
    throw "V15 trainer is already active: PID $OldPid"
}
if ((Get-PSDrive C).Free -lt 15GB) { throw "V15 resume requires at least 15GB free on C:" }
if (Test-Path -LiteralPath $StopFile) { Remove-Item -LiteralPath $StopFile -Force }

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $RunDir "train.resume.$Stamp.stdout.log"
$Stderr = Join-Path $RunDir "train.resume.$Stamp.stderr.log"
$Arguments = @(
    "scripts\train_capability_completion_v15.py",
    "--resume",
    "--checkpoint", $Checkpoint,
    "--expected-checkpoint-sha256", $CheckpointHash,
    "--dataset-dir", $DataDir,
    "--expected-manifest-sha256", $ManifestHash,
    "--output-dir", $RunDir,
    "--broad-dataset-dir", (Join-Path $Root "prepared_data\v3_broad_additional_300m_combined_v2_24k_ctx2048"),
    "--batch-size", "8",
    "--lr", "0.0000001",
    "--prompt-lm-weight", "0.05",
    "--smoke-step", "100",
    "--max-broad-relative-regression", "0.003",
    "--max-prefix-repetition-increase", "0.05",
    "--max-dead-anchor-ratio", "0.01",
    "--seed", "20260815",
    "--device", "cuda"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $RunDir "train.pid") -Encoding ascii
@{
    stage = "capability_completion_v15"
    mode = "resume"
    pid = $Process.Id
    started_at = (Get-Date).ToString("o")
    stdout = $Stdout
    stderr = $Stderr
    checkpoint = $Checkpoint
    checkpoint_sha256 = $CheckpointHash
    resume_step = 219
    resume_cursor = 1752
    dataset = $DataDir
    manifest_sha256 = $ManifestHash
    conversational_sft = $false
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunDir "active_logs.json") -Encoding utf8

Write-Output "Resumed V15 PID $($Process.Id) from step 219"
