param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Evaluator = Join-Path $Root "scripts\evaluate_base_model_strict_v3.py"
$EvaluatorHash = "AF90F4EB7F8BDA147EB9F453E0E4D96D4145D6DEA61964E0ED5C3D12A447085E"
$Checkpoint = Join-Path $Root "runs\fdt_v3_capability_completion_v20_balanced_scale_t1_20260816_r2\latest.pt"
$CheckpointHash = "7EE3F88D319928DD2D3F2542290F55FFCD036DCBB32A8AB22437C511E5890179"
$BroadDataset = Join-Path $Root "prepared_data\v3_broad_additional_300m_combined_v2_24k_ctx2048"
$CapabilityDataset = Join-Path $Root "prepared_data\capability_curriculum_5k"
$RunDir = Join-Path $Root "runs\fdt_audited_strict_v3_v20_t1_r2_20260816"
$Output = Join-Path $Root "artifacts\fdt_v20_t1_r2_strict_v3_audited_20260816.json"

$PinnedFiles = [ordered]@{
    $Evaluator = $EvaluatorHash
    $Checkpoint = $CheckpointHash
    (Join-Path $BroadDataset "manifest.json") = "BA98B6C0CF92A117076AB98A02EC70540AE4CCBFF18AB7C14BBAB5819466A885"
    (Join-Path $BroadDataset "shards\validation\source_00_shard_000000.pt") = "455978AE69641D2F5A8B25540C40C13EA6CB73FB461B1443414B5CE0C755FEA2"
    (Join-Path $BroadDataset "shards\validation\source_01_shard_000000.pt") = "1484EE2E28FCB4ACE01AEE9458E56C8F509E7FE022FB10E5C9F82DF7DDC74967"
    (Join-Path $CapabilityDataset "manifest.json") = "F2625E8FD2A9A40F9B1768EA36A7A6B8CB689276447E905EFF950FF0012EFAA9"
    (Join-Path $CapabilityDataset "shards\validation\shard_000000.pt") = "E044E8B8D2382825994A1115FC95A5553ADF556DE503CB4DBA83BFC6CDC1FFE6"
}

foreach ($Path in @($Root, $RunDir, $Output) + @($PinnedFiles.Keys)) {
    $FullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $FullPath.StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "V20 T1 R2 audited evaluation permits only C: paths: $FullPath"
    }
}
foreach ($Item in $PinnedFiles.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $Item.Key)) {
        throw "Missing pinned input: $($Item.Key)"
    }
    if ((Get-FileHash -LiteralPath $Item.Key -Algorithm SHA256).Hash -ne $Item.Value) {
        throw "Pinned input hash mismatch: $($Item.Key)"
    }
}
if (Test-Path -LiteralPath $Output) {
    throw "Audited output already exists: $Output"
}
if (Test-Path -LiteralPath $RunDir) {
    if (Get-ChildItem -LiteralPath $RunDir -Force) {
        throw "Audited run directory is not empty: $RunDir"
    }
} else {
    New-Item -ItemType Directory -Path $RunDir | Out-Null
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $RunDir "eval.$Stamp.stdout.log"
$Stderr = Join-Path $RunDir "eval.$Stamp.stderr.log"
$Arguments = @(
    "scripts\evaluate_base_model_strict_v3.py",
    "--checkpoint", $Checkpoint,
    "--dataset_dir", $BroadDataset,
    "--capability_dataset_dir", $CapabilityDataset,
    "--output", $Output,
    "--mode", "standard",
    "--batch_size", "4",
    "--device", "cuda"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $RunDir "eval.pid") -Encoding ascii
@{
    stage = "audited_strict_v3_v20_t1_r2"
    pid = $Process.Id
    started_at = (Get-Date).ToString("o")
    stdout = $Stdout
    stderr = $Stderr
    evaluator = $Evaluator
    evaluator_sha256 = $EvaluatorHash
    checkpoint = $Checkpoint
    checkpoint_sha256 = $CheckpointHash
    broad_validation = $BroadDataset
    capability_validation = $CapabilityDataset
    input_pins = $PinnedFiles
    output = $Output
    mode = "standard"
    batch_size = 4
    device = "cuda"
    training = $false
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $RunDir "active_logs.json") -Encoding utf8

Write-Output "Started audited strict-v3 V20 T1 R2 evaluation PID $($Process.Id)"
