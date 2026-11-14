$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$run = Join-Path $root "runs\fdt_v3_scaled_1216_20l_scratch_v2_300m"
$config = Join-Path $root "configs\fdt_v3_scaled_1216_20l_scratch_300m.yaml"
$checkpoint = Join-Path $run "latest.pt"
$trainer = Join-Path $root "scripts\train_next_model.py"
$stopFile = Join-Path $run "STOP_REQUESTED"

foreach ($path in @($run, $config, $checkpoint, $trainer)) {
    $full = [System.IO.Path]::GetFullPath($path)
    if (-not $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside the C-only workspace: $full"
    }
    if ([System.IO.Path]::GetPathRoot($full) -ne "C:\") {
        throw "Refusing non-C storage path: $full"
    }
}

if (-not (Test-Path -LiteralPath $checkpoint)) {
    throw "Missing resume checkpoint: $checkpoint"
}
if (Test-Path -LiteralPath $stopFile) {
    throw "STOP_REQUESTED still exists: $stopFile"
}

$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.CommandLine -match 'train_next_model.py' -and
    $_.CommandLine -match 'fdt_v3_scaled_1216_20l_scratch'
}
if ($active) {
    throw "A scaled FDT trainer is already active: $($active.ProcessId -join ', ')"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $run "train.$stamp.stdout.log"
$stderr = Join-Path $run "train.$stamp.stderr.log"
$python = (Get-Command python).Source
$arguments = @(
    "-u",
    $trainer,
    "--config", $config,
    "--resume_from", $checkpoint
)

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
    resumed_from = $checkpoint
    stdout = $stdout
    stderr = $stderr
    config = $config
    graceful_stop_file = $stopFile
    storage_drive = "C:"
    lr_decay_steps = 12207
    eval_sampling = "evenly_spaced"
    eval_batches = 20
}
$activeLogs | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run "active_logs.json") -Encoding utf8

Start-Sleep -Seconds 3
if ($process.HasExited) {
    $errorTail = if (Test-Path -LiteralPath $stderr) {
        Get-Content -LiteralPath $stderr -Tail 40 | Out-String
    } else {
        "No stderr log was created."
    }
    throw "Trainer exited during launch. $errorTail"
}

$activeLogs | ConvertTo-Json
