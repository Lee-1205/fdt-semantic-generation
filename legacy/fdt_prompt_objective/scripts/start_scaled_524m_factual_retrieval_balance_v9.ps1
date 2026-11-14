$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$run = Join-Path $root "runs\fdt_v3_scaled_1216_20l_550m_general_growth_v6_scale"
$config = Join-Path $root "configs\fdt_v3_scaled_1216_20l_524m_factual_retrieval_balance_v9.yaml"
$dataset = Join-Path $root "prepared_data\v3_factual_retrieval_balance_v9_1500k_ctx2048"
$trainer = Join-Path $root "scripts\train_next_model.py"
$checkpoint = Join-Path $run "latest.pt"
$stopFile = Join-Path $run "STOP_REQUESTED"
$expectedCheckpointHash = "36A40B569465F3688C2B5FDDCB3024B14FB7B4DCC2E6B0BED113DD3D76B14B1B"
$expectedDatasetManifestHash = "8CE8D58EB6950D83577CAEA7D5B2D8F3721F1D10603DD1F0527E3095C4AD5551"

foreach ($path in @($run, $config, $dataset, $trainer, $checkpoint)) {
    $full = [System.IO.Path]::GetFullPath($path)
    if (-not $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside the C-only workspace: $full"
    }
    if ([System.IO.Path]::GetPathRoot($full) -ne "C:\") {
        throw "Refusing non-C storage path: $full"
    }
}
foreach ($path in @($config, $dataset, $trainer, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required input: $path"
    }
}
if (Test-Path -LiteralPath $stopFile) {
    throw "STOP_REQUESTED still exists: $stopFile"
}
$actualCheckpointHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $checkpoint).Hash
if ($actualCheckpointHash -ne $expectedCheckpointHash) {
    throw "523M checkpoint hash mismatch: $actualCheckpointHash"
}
$actualDatasetManifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $dataset "manifest.json")).Hash
if ($actualDatasetManifestHash -ne $expectedDatasetManifestHash) {
    throw "Dataset manifest hash mismatch: $actualDatasetManifestHash"
}

$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'train_next_model.py'
}
if ($active) {
    throw "Another next-model trainer is already active: $($active.ProcessId -join ', ')"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $run "train.balance_v9.$stamp.stdout.log"
$stderr = Join-Path $run "train.balance_v9.$stamp.stderr.log"
$python = (Get-Command python).Source
$arguments = @("-u", $trainer, "--config", $config, "--resume_from", $checkpoint)

$env:PYTHONUNBUFFERED = "1"
$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $root `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

Set-Content -LiteralPath (Join-Path $run "train.pid") -Value $process.Id -Encoding ascii
$activeLogs = [ordered]@{
    started_at = (Get-Date).ToString("o")
    pid = $process.Id
    stage = "factual_retrieval_balance_v9_1500k"
    resumed_from = $checkpoint
    parent_sha256 = $actualCheckpointHash
    stdout = $stdout
    stderr = $stderr
    config = $config
    dataset = $dataset
    dataset_manifest_sha256 = $actualDatasetManifestHash
    graceful_stop_file = $stopFile
    storage_drive = "C:"
    target_optimizer_steps = 91
    target_global_step = 32009
    target_tokens = 524435456
    learning_rate = 0.0000002
}
$activeLogs | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run "active_logs.json") -Encoding utf8

Start-Sleep -Seconds 3
if ($process.HasExited) {
    $errorTail = if (Test-Path -LiteralPath $stderr) {
        Get-Content -LiteralPath $stderr -Tail 80 | Out-String
    } else {
        "No stderr log was created."
    }
    throw "Trainer exited during launch. $errorTail"
}

$activeLogs | ConvertTo-Json
