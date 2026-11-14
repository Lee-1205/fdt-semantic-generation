$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$run = Join-Path $root "runs\fdt_v3_scaled_1216_20l_502m_retention_capability_v2_pilot"
$config = Join-Path $root "configs\fdt_v3_scaled_1216_20l_502m_retention_capability_v2_pilot.yaml"
$dataset = Join-Path $root "prepared_data\v3_retention_capability_v2_2m_pilot_ctx2048_r2"
$trainer = Join-Path $root "scripts\train_next_model.py"
$ownCheckpoint = Join-Path $run "latest.pt"
$frozenCheckpoint = Join-Path $root "runs\fdt_v3_scaled_1216_20l_scratch_v2_500m\latest.pt"
$checkpoint = if (Test-Path -LiteralPath $ownCheckpoint) { $ownCheckpoint } else { $frozenCheckpoint }
$stopFile = Join-Path $run "STOP_REQUESTED"
$expectedBaseHash = "7359C47484E86EFF88B5EAF05CE8B36345A56BFEF888C5FB50B3CB8221829683"

$run = [System.IO.Path]::GetFullPath($run)
New-Item -ItemType Directory -Force -Path $run | Out-Null
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
if (-not (Test-Path -LiteralPath (Join-Path $dataset "manifest.json"))) {
    throw "Dataset manifest is missing"
}
if (-not (Test-Path -LiteralPath $ownCheckpoint)) {
    $actualBaseHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $frozenCheckpoint).Hash
    if ($actualBaseHash -ne $expectedBaseHash) {
        throw "Frozen 500M checkpoint hash mismatch: $actualBaseHash"
    }
}
if (Test-Path -LiteralPath $stopFile) {
    throw "STOP_REQUESTED still exists: $stopFile"
}

$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'train_next_model.py'
}
if ($active) {
    throw "Another next-model trainer is already active: $($active.ProcessId -join ', ')"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $run "train.$stamp.stdout.log"
$stderr = Join-Path $run "train.$stamp.stderr.log"
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
    stage = "retention_capability_v2_2m_pilot"
    resumed_from = $checkpoint
    stdout = $stdout
    stderr = $stderr
    config = $config
    dataset = $dataset
    graceful_stop_file = $stopFile
    storage_drive = "C:"
    target_optimizer_steps = 121
    target_tokens = 501989376
    learning_rate = 0.0000015
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
