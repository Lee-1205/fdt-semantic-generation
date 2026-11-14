param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Baseline = Join-Path $Root "runs\fdt_v3_scaled_1216_20l_520m_retention_alignment_v4_pilot\latest.pt"
$Candidate = Join-Path $Root "runs\fdt_v3_capability_completion_v15_r2\latest.pt"
$Dataset = Join-Path $Root "prepared_data\v3_broad_additional_300m_combined_v2_24k_ctx2048"
$RunDir = Join-Path $Root "runs\fdt_v15_expanded_repetition_eval_20260813_r2"
$Output = Join-Path $Root "artifacts\fdt_v15_expanded_repetition_eval_20260813_r2.json"
$BaselineHash = "FC758EF3028734E31B763019B8F96DFCC341AD5120D9751BF1211920E9AA1D59"
$CandidateHash = "E8D5C00CAEC9E13D4A5BB78B5DDA15A94B7AA8497D37CF0C3586F148FBAF47CE"

foreach ($Path in @($Root, $Baseline, $Candidate, $Dataset, $RunDir, $Output)) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Expanded repetition evaluation permits only C: paths: $Path"
    }
}
if ((Get-FileHash -LiteralPath $Baseline -Algorithm SHA256).Hash -ne $BaselineHash) {
    throw "Promoted 520M hash mismatch"
}
if ((Get-FileHash -LiteralPath $Candidate -Algorithm SHA256).Hash -ne $CandidateHash) {
    throw "Preserved V15 hash mismatch"
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
    stage = "v15_expanded_repetition_eval_r2"
    pid = $Process.Id
    started_at = (Get-Date).ToString("o")
    stdout = $Stdout
    stderr = $Stderr
    baseline = $Baseline
    baseline_sha256 = $BaselineHash
    candidate = $Candidate
    candidate_sha256 = $CandidateHash
    output = $Output
    partial_output = "$Output.partial"
    training = $false
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunDir "active_logs.json") -Encoding utf8

Write-Output "Started expanded V15 repetition evaluation R2 PID $($Process.Id)"
