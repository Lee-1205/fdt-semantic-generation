param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Parent = Join-Path $Root "runs\fdt_v3_scaled_1216_20l_520m_retention_alignment_v4_pilot\latest.pt"
$ParentHash = "FC758EF3028734E31B763019B8F96DFCC341AD5120D9751BF1211920E9AA1D59"
$DataDir = Join-Path $Root "prepared_data\v3_capability_completion_v15_5k_ctx512"
$ManifestHash = "97FC204DA290100E0CEF1E7B542276CECB30E8894492774B9FDE10BCEED62A85"
$RunDir = Join-Path $Root "runs\fdt_v3_capability_completion_v15_r2"

foreach ($Path in @($Root, $Parent, $DataDir, $RunDir)) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "V15 permits only C: paths: $Path"
    }
}
if (-not (Test-Path -LiteralPath $Parent)) { throw "Missing promoted 520M parent" }
if (-not (Test-Path -LiteralPath (Join-Path $DataDir "manifest.json"))) { throw "Missing V15 manifest" }
if ((Get-FileHash -LiteralPath $Parent -Algorithm SHA256).Hash -ne $ParentHash) {
    throw "Promoted 520M parent hash mismatch"
}
if ((Get-FileHash -LiteralPath (Join-Path $DataDir "manifest.json") -Algorithm SHA256).Hash -ne $ManifestHash) {
    throw "V15 manifest hash mismatch"
}
if ((Get-PSDrive C).Free -lt 15GB) { throw "V15 requires at least 15GB free on C:" }
if (Test-Path -LiteralPath $RunDir) {
    if (Get-ChildItem -LiteralPath $RunDir -Force) { throw "Dedicated V15 output is not empty" }
} else {
    New-Item -ItemType Directory -Path $RunDir | Out-Null
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $RunDir "train.$Stamp.stdout.log"
$Stderr = Join-Path $RunDir "train.$Stamp.stderr.log"
$Arguments = @(
    "scripts\train_capability_completion_v15.py",
    "--checkpoint", $Parent,
    "--expected-checkpoint-sha256", $ParentHash,
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
    "--device", "cuda"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $RunDir "train.pid") -Encoding ascii
@{
    stage = "capability_completion_v15"
    pid = $Process.Id
    started_at = (Get-Date).ToString("o")
    stdout = $Stdout
    stderr = $Stderr
    checkpoint = $Parent
    checkpoint_sha256 = $ParentHash
    dataset = $DataDir
    manifest_sha256 = $ManifestHash
    conversational_sft = $false
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunDir "active_logs.json") -Encoding utf8

Write-Output "Started V15 PID $($Process.Id)"
