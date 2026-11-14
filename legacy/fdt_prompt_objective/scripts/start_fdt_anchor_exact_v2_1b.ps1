$ErrorActionPreference = "Stop"

$Root = "<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM"
$Parent = Join-Path $Root "runs\fdt_v21_t2_step1000_interp_a025_20260821_r2\latest.pt"
$Output = Join-Path $Root "runs\fdt_anchor_exact_v2_1b_20260821"
$Tokenizer = Join-Path $Root "tokenizers\fdt_v3_bpe_24k"
$Validation = Join-Path $Root "prepared_data\v3_knowledge_scale_v21_t3_100k_ctx512_20260821\shards\validation\shard_000000.pt"
$Trainer = Join-Path $Root "scripts\train_fdt_dual_memory_streaming.py"
$Server = Join-Path $Root "apps\fdt_anchor_exact_console\server.py"
$StateDir = Join-Path $Root "runs\fdt_anchor_exact_console"

foreach ($Path in @($Root, $Parent, $Output, $Tokenizer, $Validation, $Trainer, $Server, $StateDir)) {
    if ([IO.Path]::GetFullPath($Path).Substring(0, 2).ToUpperInvariant() -ne "C:") {
        throw "Only C: paths are permitted: $Path"
    }
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Parent).Hash -ne "A4EDBF7E02AFE0AF3388220D3934BCBA171B8A1942D83BAA8EED59C654BB341E") {
    throw "Parent checkpoint hash mismatch"
}
if (Test-Path -LiteralPath $Output) {
    throw "Fresh output already exists: $Output"
}
New-Item -ItemType Directory -Path $Output | Out-Null
New-Item -ItemType Directory -Path $StateDir -Force | Out-Null

$Stdout = Join-Path $Output "stdout.log"
$Stderr = Join-Path $Output "stderr.log"
$Arguments = @(
    "-u", $Trainer,
    "--parent", $Parent,
    "--model-type", "fdt_v3_anchor_indexed_dual_memory",
    "--output-dir", $Output,
    "--tokenizer", $Tokenizer,
    "--target-additional-tokens", "1000000000",
    "--batch-size", "2", "--grad-accum", "5",
    "--learning-rate", "2.5e-9", "--pointer-loss-weight", "0.05",
    "--save-every", "999999999", "--initial-save-step", "999999999",
    "--recovery-every", "1221", "--initial-recovery-step", "100",
    "--milestone-tokens", "100000000",
    "--validation-shard", $Validation,
    "--overfit-check-every", "5000", "--overfit-relative-regression", "0.005",
    "--overfit-patience", "3", "--log-every", "1",
    "--seed", "20261021", "--min-free-gib", "10", "--device", "cuda"
)
$env:PYTHONPATH = Join-Path $Root "src"
$env:CUDA_MODULE_LOADING = "LAZY"
$Training = Start-Process -FilePath "python" -ArgumentList $Arguments -WorkingDirectory $Root -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Training.Id | Set-Content -LiteralPath (Join-Path $Output "train.pid") -Encoding ascii
@{pid=$Training.Id;output=$Output;stdout=$Stdout;stderr=$Stderr;target_additional_tokens=1000000000} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output "active_logs.json") -Encoding utf8

$PortOwner = Get-NetTCPConnection -LocalPort 7861 -State Listen -ErrorAction SilentlyContinue
if ($PortOwner) {
    throw "Port 7861 is already in use"
}
$ConsoleOut = Join-Path $StateDir "stdout.log"
$ConsoleErr = Join-Path $StateDir "stderr.log"
$Console = Start-Process -FilePath "python" -ArgumentList @("-u", $Server, "--port", "7861", "--run-dir", $Output, "--trainer-pid", $Training.Id) -WorkingDirectory $Root -RedirectStandardOutput $ConsoleOut -RedirectStandardError $ConsoleErr -WindowStyle Hidden -PassThru
$Console.Id | Set-Content -LiteralPath (Join-Path $StateDir "server.pid") -Encoding ascii
Write-Output "FDT Anchor Exact V2 training PID $($Training.Id); console PID $($Console.Id)"
