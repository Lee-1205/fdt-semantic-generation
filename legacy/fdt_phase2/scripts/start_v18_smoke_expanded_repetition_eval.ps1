param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Baseline = Join-Path $Root "runs\fdt_v3_capability_completion_v17_balanced_20260814\latest.pt"
$Candidate = Join-Path $Root "runs\fdt_v3_capability_completion_v18_calibrated_retention_20260814_r2\latest.pt"
$Dataset = Join-Path $Root "prepared_data\v3_broad_additional_300m_combined_v2_24k_ctx2048"
$Evaluator = Join-Path $Root "scripts\evaluate_v15_expanded_repetition.py"
$Scoring = Join-Path $Root "scripts\evaluate_base_model_strict_v3.py"
$RunDir = Join-Path $Root "runs\fdt_v18_smoke_expanded_repetition_eval_20260814"
$Output = Join-Path $Root "artifacts\fdt_v18_smoke_expanded_repetition_eval_20260814.json"

$Pinned = @{
    $Baseline = "A7C418B6014412E39D6C9C659C7614A0677D7D062CE1A6A7239A7DD8CF0A8D92"
    $Candidate = "B2A8B84C42496D8AF05AC1531F7DCA34EE031CB059C14033F7F18FDDF086CFDA"
    $Evaluator = "DF8B79F3A31A50F9965AE01036D44873EB94923C219F62BD1CC9F0EEA3EEDC08"
    $Scoring = "AF90F4EB7F8BDA147EB9F453E0E4D96D4145D6DEA61964E0ED5C3D12A447085E"
}
foreach ($Path in @($Root, $Baseline, $Candidate, $Dataset, $Evaluator, $Scoring, $RunDir, $Output)) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Expanded repetition evaluation permits only C: paths: $Path"
    }
}
foreach ($Item in $Pinned.GetEnumerator()) {
    if ((Get-FileHash -LiteralPath $Item.Key -Algorithm SHA256).Hash -ne $Item.Value) {
        throw "Pinned input hash mismatch: $($Item.Key)"
    }
}
if (Test-Path -LiteralPath $Output) { throw "Expanded repetition output already exists" }
if (Test-Path -LiteralPath $RunDir) {
    if (Get-ChildItem -LiteralPath $RunDir -Force) { throw "Expanded repetition run directory is not empty" }
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
    "--seed", "20260818",
    "--device", "cuda"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $RunDir "eval.pid") -Encoding ascii
@{
    stage = "v18_smoke_expanded_repetition_eval"
    pid = $Process.Id
    started_at = (Get-Date).ToString("o")
    stdout = $Stdout
    stderr = $Stderr
    baseline = $Baseline
    baseline_sha256 = $Pinned[$Baseline]
    candidate = $Candidate
    candidate_sha256 = $Pinned[$Candidate]
    evaluator_sha256 = $Pinned[$Evaluator]
    scoring_sha256 = $Pinned[$Scoring]
    output = $Output
    partial_output = "$Output.partial"
    training = $false
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunDir "active_logs.json") -Encoding utf8

Write-Output "Started V18 smoke expanded repetition evaluation PID $($Process.Id)"
