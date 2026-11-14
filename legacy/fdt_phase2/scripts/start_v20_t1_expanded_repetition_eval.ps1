param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Baseline = Join-Path $Root "runs\fdt_v3_capability_completion_v18_calibrated_retention_20260815_r4_resume\latest.pt"
$Candidate = Join-Path $Root "runs\fdt_v3_capability_completion_v20_balanced_scale_t1_20260816_r2\latest.pt"
$Dataset = Join-Path $Root "prepared_data\v3_broad_additional_300m_combined_v2_24k_ctx2048"
$Evaluator = Join-Path $Root "scripts\evaluate_v15_expanded_repetition.py"
$ScoringModule = Join-Path $Root "scripts\evaluate_base_model_strict_v3.py"
$RunDir = Join-Path $Root "runs\fdt_v20_t1_expanded_repetition_eval_20260816"
$Output = Join-Path $Root "artifacts\fdt_v20_t1_expanded_repetition_vs_v18_20260816.json"

$Pinned = @{
    $Baseline = "23C5539B7EF870311DB36E65F21896343A8F53D5203F603E83DD2F5CD02745B6"
    $Candidate = "7EE3F88D319928DD2D3F2542290F55FFCD036DCBB32A8AB22437C511E5890179"
    $Evaluator = "DF8B79F3A31A50F9965AE01036D44873EB94923C219F62BD1CC9F0EEA3EEDC08"
    $ScoringModule = "AF90F4EB7F8BDA147EB9F453E0E4D96D4145D6DEA61964E0ED5C3D12A447085E"
}
foreach ($Path in @($Root, $Baseline, $Candidate, $Dataset, $Evaluator, $ScoringModule, $RunDir, $Output)) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Expanded repetition evaluation permits only C: paths: $Path"
    }
}
foreach ($Item in $Pinned.GetEnumerator()) {
    if ((Get-FileHash -LiteralPath $Item.Key -Algorithm SHA256).Hash -ne $Item.Value) {
        throw "Pinned input hash mismatch: $($Item.Key)"
    }
}
if (Test-Path -LiteralPath $Output) { throw "Expanded evaluation artifact already exists" }
if (Test-Path -LiteralPath $RunDir) {
    if (Get-ChildItem -LiteralPath $RunDir -Force) { throw "Expanded evaluation run directory is not empty" }
} else {
    New-Item -ItemType Directory -Path $RunDir | Out-Null
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $RunDir "eval.$Stamp.stdout.log"
$Stderr = Join-Path $RunDir "eval.$Stamp.stderr.log"
$Arguments = @(
    "scripts\evaluate_v15_expanded_repetition.py",
    "--baseline", $Baseline,
    "--candidate", $Candidate,
    "--dataset-dir", $Dataset,
    "--output", $Output,
    "--rows-per-bucket", "20",
    "--exclude-standard-rows", "10",
    "--prefix-tokens", "96",
    "--continuation-tokens", "64",
    "--bootstrap-samples", "100000",
    "--seed", "20260813",
    "--device", "cuda"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $RunDir "eval.pid") -Encoding ascii
@{
    stage = "v20_t1_expanded_repetition_eval"
    pid = $Process.Id
    started_at = (Get-Date).ToString("o")
    stdout = $Stdout
    stderr = $Stderr
    baseline = $Baseline
    baseline_sha256 = $Pinned[$Baseline]
    candidate = $Candidate
    candidate_sha256 = $Pinned[$Candidate]
    evaluator_sha256 = $Pinned[$Evaluator]
    scoring_module_sha256 = $Pinned[$ScoringModule]
    output = $Output
    partial_output = "$Output.partial"
    training = $false
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunDir "active_logs.json") -Encoding utf8

Write-Output "Started expanded V20 tranche-1 repetition evaluation PID $($Process.Id)"
