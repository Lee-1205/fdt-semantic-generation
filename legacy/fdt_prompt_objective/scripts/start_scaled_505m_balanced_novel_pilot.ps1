$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$run = Join-Path $root "runs\fdt_v3_scaled_1216_20l_505m_balanced_novel_pilot"
$config = Join-Path $root "configs\fdt_v3_scaled_1216_20l_505m_balanced_novel_pilot.yaml"
$baseCheckpoint = Join-Path $root "runs\fdt_v3_scaled_1216_20l_scratch_v2_500m\latest.pt"
$runCheckpoint = Join-Path $run "latest.pt"
$dataset = Join-Path $root "prepared_data\v3_balanced_novel_5m_pilot_ctx2048"
$manifest = Join-Path $dataset "manifest.json"
$trainer = Join-Path $root "scripts\train_next_model.py"
$stopFile = Join-Path $run "STOP_REQUESTED"
$expectedBaseSha = "7359C47484E86EFF88B5EAF05CE8B36345A56BFEF888C5FB50B3CB8221829683"
$expectedManifestSha = "3B5EF79592FBC401408510D110DE2FDDDA58B53266E245534E5873851A1D8CE6"

$run = [System.IO.Path]::GetFullPath($run)
New-Item -ItemType Directory -Force -Path $run | Out-Null
foreach ($path in @($run, $config, $baseCheckpoint, $dataset, $manifest, $trainer)) {
    $full = [System.IO.Path]::GetFullPath($path)
    if (-not $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside the C-only workspace: $full"
    }
    if ([System.IO.Path]::GetPathRoot($full) -ne "C:\") {
        throw "Refusing non-C storage path: $full"
    }
}

foreach ($path in @($config, $baseCheckpoint, $dataset, $manifest, $trainer)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required input: $path"
    }
}
if ((Get-FileHash -LiteralPath $baseCheckpoint -Algorithm SHA256).Hash -ne $expectedBaseSha) {
    throw "Frozen 500M checkpoint hash mismatch"
}
if ((Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash -ne $expectedManifestSha) {
    throw "Balanced-novel dataset manifest hash mismatch"
}
if (Test-Path -LiteralPath $stopFile) {
    throw "STOP_REQUESTED still exists: $stopFile"
}

$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.CommandLine -match 'train_next_model.py' -and
    $_.CommandLine -match 'fdt_v3_scaled_1216_20l'
}
if ($active) {
    throw "A scaled FDT trainer is already active: $($active.ProcessId -join ', ')"
}
if (Get-ScheduledTask -TaskName "GoogleUpdateTaskSYSTEM" -ErrorAction SilentlyContinue) {
    throw "Security stop indicator GoogleUpdateTaskSYSTEM is present"
}

$resumeCheckpoint = if (Test-Path -LiteralPath $runCheckpoint) {
    $runCheckpoint
} else {
    $baseCheckpoint
}
$resumeSha = (Get-FileHash -LiteralPath $resumeCheckpoint -Algorithm SHA256).Hash
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $run "train.$stamp.stdout.log"
$stderr = Join-Path $run "train.$stamp.stderr.log"
$python = (Get-Command python).Source
$arguments = @(
    "-u",
    $trainer,
    "--config", $config,
    "--resume_from", $resumeCheckpoint
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
    stage = "505m_balanced_novel_5m_pilot"
    resumed_from = $resumeCheckpoint
    resumed_from_sha256 = $resumeSha
    dataset = $dataset
    dataset_manifest_sha256 = $expectedManifestSha
    stdout = $stdout
    stderr = $stderr
    config = $config
    graceful_stop_file = $stopFile
    storage_drive = "C:"
    learning_rate = 0.000003
    lr_schedule = "constant"
    target_tokens = 505004032
    eval_sampling = "evenly_spaced"
    eval_batches = 20
}
$activeLogs | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run "active_logs.json") -Encoding utf8

Start-Sleep -Seconds 4
if ($process.HasExited) {
    $errorTail = if (Test-Path -LiteralPath $stderr) {
        Get-Content -LiteralPath $stderr -Tail 60 | Out-String
    } else {
        "No stderr log was created."
    }
    throw "Trainer exited during launch. $errorTail"
}

$activeLogs | ConvertTo-Json
