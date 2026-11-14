$ErrorActionPreference = "Stop"

$Root = "<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM"
$Parent = Join-Path $Root "runs\fdt_v21_t2_step1000_interp_a025_20260821_r2\latest.pt"
$Resume = Join-Path $Root "runs\fdt_dual_memory_v1_1b_20260821_r5_metrics_resume\latest_recovery.pt"
$Output = Join-Path $Root "runs\fdt_dual_memory_v1_1b_20260821_r6_live_metrics_resume"
$Tokenizer = Join-Path $Root "tokenizers\fdt_v3_bpe_24k"
$Validation = Join-Path $Root "prepared_data\v3_knowledge_scale_v21_t3_100k_ctx512_20260821\shards\validation\shard_000000.pt"
$Trainer = Join-Path $Root "scripts\train_fdt_dual_memory_streaming.py"
$Monitor = Join-Path $Root "apps\fdt_dual_memory_status.py"

foreach ($Path in @($Root, $Parent, $Resume, $Output, $Tokenizer, $Trainer, $Validation, $Monitor)) {
    if ([IO.Path]::GetFullPath($Path).Substring(0, 2).ToUpperInvariant() -ne "C:") {
        throw "Only C: paths are permitted: $Path"
    }
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Parent).Hash -ne "A4EDBF7E02AFE0AF3388220D3934BCBA171B8A1942D83BAA8EED59C654BB341E") {
    throw "Parent checkpoint hash mismatch"
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Resume).Hash -ne "A628C3BDE3D8D9EE2AECECF8AD2D4B886380D91BB43D54552EB18E183514D08D") {
    throw "Resume checkpoint hash mismatch"
}
if (Test-Path -LiteralPath $Output) {
    throw "Fresh output already exists: $Output"
}
New-Item -ItemType Directory -Path $Output | Out-Null

$Stdout = Join-Path $Output "stdout.log"
$Stderr = Join-Path $Output "stderr.log"
$Arguments = @(
    "-u", $Trainer,
    "--parent", $Parent,
    "--resume", $Resume,
    "--output-dir", $Output,
    "--tokenizer", $Tokenizer,
    "--target-additional-tokens", "1000000000",
    "--batch-size", "2",
    "--grad-accum", "5",
    "--learning-rate", "2.5e-9",
    "--pointer-loss-weight", "0.05",
    "--save-every", "250",
    "--initial-save-step", "25",
    "--recovery-every", "1000",
    "--initial-recovery-step", "100",
    "--validation-shard", $Validation,
    "--overfit-check-every", "500",
    "--overfit-relative-regression", "0.005",
    "--overfit-patience", "3",
    "--log-every", "1",
    "--seed", "20260921",
    "--min-free-gib", "10",
    "--device", "cuda"
)
$env:PYTHONPATH = Join-Path $Root "src"
$env:CUDA_MODULE_LOADING = "LAZY"
$Process = Start-Process -FilePath "python" -ArgumentList $Arguments -WorkingDirectory $Root -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $Output "train.pid") -Encoding ascii
@{
    pid = $Process.Id
    output = $Output
    stdout = $Stdout
    stderr = $Stderr
    resume = $Resume
    target_additional_tokens = 1000000000
    evaluation_during_training = $false
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output "active_logs.json") -Encoding utf8

$Pythonw = (Get-Command pythonw).Source
$MonitorProcess = Start-Process -FilePath $Pythonw -ArgumentList @($Monitor, "--run-dir", $Output) -WorkingDirectory $Root -PassThru
$MonitorProcess.Id | Set-Content -LiteralPath (Join-Path $Output "monitor.pid") -Encoding ascii
Write-Output "Resumed FDT Dual Memory training (PID $($Process.Id)); status window PID $($MonitorProcess.Id)"
