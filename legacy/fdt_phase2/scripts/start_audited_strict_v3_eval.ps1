param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("520m", "v15", "v16")]
    [string]$Target,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Evaluator = Join-Path $Root "scripts\evaluate_base_model_strict_v3.py"
$EvaluatorHash = "AF90F4EB7F8BDA147EB9F453E0E4D96D4145D6DEA61964E0ED5C3D12A447085E"
$BroadDataset = Join-Path $Root "prepared_data\v3_broad_additional_300m_combined_v2_24k_ctx2048"
$CapabilityDataset = Join-Path $Root "prepared_data\capability_curriculum_5k"

$Targets = @{
    "520m" = @{
        checkpoint = Join-Path $Root "runs\fdt_v3_scaled_1216_20l_520m_retention_alignment_v4_pilot\latest.pt"
        checkpoint_sha256 = "FC758EF3028734E31B763019B8F96DFCC341AD5120D9751BF1211920E9AA1D59"
        run_dir = Join-Path $Root "runs\fdt_audited_strict_v3_520m_20260813_r2"
        output = Join-Path $Root "artifacts\fdt_520m_strict_v3_audited_20260813_r2.json"
    }
    "v15" = @{
        checkpoint = Join-Path $Root "runs\fdt_v3_capability_completion_v15_r2\latest.pt"
        checkpoint_sha256 = "E8D5C00CAEC9E13D4A5BB78B5DDA15A94B7AA8497D37CF0C3586F148FBAF47CE"
        run_dir = Join-Path $Root "runs\fdt_audited_strict_v3_v15_20260813_r2"
        output = Join-Path $Root "artifacts\fdt_v15_strict_v3_audited_20260813_r2.json"
    }
    "v16" = @{
        checkpoint = Join-Path $Root "runs\fdt_v3_capability_completion_v16_v15_parent_capability_heavy_20260813\latest.pt"
        checkpoint_sha256 = "7883BA4EC7AEE545B2CA9CBB395F2A357166811C7070C688942210F4B8864FF2"
        run_dir = Join-Path $Root "runs\fdt_audited_strict_v3_v16_20260813_r2"
        output = Join-Path $Root "artifacts\fdt_v16_strict_v3_audited_20260813_r2.json"
    }
}

$PinnedFiles = @{
    (Join-Path $BroadDataset "manifest.json") = "BA98B6C0CF92A117076AB98A02EC70540AE4CCBFF18AB7C14BBAB5819466A885"
    (Join-Path $BroadDataset "shards\validation\source_00_shard_000000.pt") = "455978AE69641D2F5A8B25540C40C13EA6CB73FB461B1443414B5CE0C755FEA2"
    (Join-Path $BroadDataset "shards\validation\source_01_shard_000000.pt") = "1484EE2E28FCB4ACE01AEE9458E56C8F509E7FE022FB10E5C9F82DF7DDC74967"
    (Join-Path $CapabilityDataset "manifest.json") = "F2625E8FD2A9A40F9B1768EA36A7A6B8CB689276447E905EFF950FF0012EFAA9"
    (Join-Path $CapabilityDataset "shards\validation\shard_000000.pt") = "E044E8B8D2382825994A1115FC95A5553ADF556DE503CB4DBA83BFC6CDC1FFE6"
}

$Spec = $Targets[$Target]
$AllPaths = @(
    $Root,
    $Evaluator,
    $BroadDataset,
    $CapabilityDataset,
    $Spec.checkpoint,
    $Spec.run_dir,
    $Spec.output
) + @($PinnedFiles.Keys)
foreach ($Path in $AllPaths) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Audited strict evaluation permits only C: paths: $Path"
    }
}

if ((Get-FileHash -LiteralPath $Evaluator -Algorithm SHA256).Hash -ne $EvaluatorHash) {
    throw "Audited evaluator hash mismatch"
}
if ((Get-FileHash -LiteralPath $Spec.checkpoint -Algorithm SHA256).Hash -ne $Spec.checkpoint_sha256) {
    throw "$Target checkpoint hash mismatch"
}
foreach ($Item in $PinnedFiles.GetEnumerator()) {
    if ((Get-FileHash -LiteralPath $Item.Key -Algorithm SHA256).Hash -ne $Item.Value) {
        throw "Validation input hash mismatch: $($Item.Key)"
    }
}
if (Test-Path -LiteralPath $Spec.output) {
    throw "Audited output already exists: $($Spec.output)"
}
if (Test-Path -LiteralPath $Spec.run_dir) {
    if (Get-ChildItem -LiteralPath $Spec.run_dir -Force) {
        throw "Audited run directory is not empty: $($Spec.run_dir)"
    }
} else {
    New-Item -ItemType Directory -Path $Spec.run_dir | Out-Null
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $Spec.run_dir "eval.$Stamp.stdout.log"
$Stderr = Join-Path $Spec.run_dir "eval.$Stamp.stderr.log"
$Arguments = @(
    "scripts\evaluate_base_model_strict_v3.py",
    "--checkpoint", $Spec.checkpoint,
    "--dataset_dir", $BroadDataset,
    "--capability_dataset_dir", $CapabilityDataset,
    "--output", $Spec.output,
    "--mode", "standard",
    "--batch_size", "4",
    "--device", "cuda"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $Spec.run_dir "eval.pid") -Encoding ascii
@{
    stage = "audited_strict_v3_$Target"
    pid = $Process.Id
    started_at = (Get-Date).ToString("o")
    stdout = $Stdout
    stderr = $Stderr
    evaluator = $Evaluator
    evaluator_sha256 = $EvaluatorHash
    checkpoint = $Spec.checkpoint
    checkpoint_sha256 = $Spec.checkpoint_sha256
    broad_validation = $BroadDataset
    capability_validation = $CapabilityDataset
    output = $Spec.output
    mode = "standard"
    batch_size = 4
    device = "cuda"
    training = $false
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Spec.run_dir "active_logs.json") -Encoding utf8

Write-Output "Started audited strict-v3 $Target evaluation PID $($Process.Id)"
