$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$run = Join-Path $root "runs\fdt_v3_scaled_1216_20l_520m_retention_alignment_v4_pilot"
$config = Join-Path $root "configs\fdt_v3_scaled_1216_20l_520m_retention_alignment_v4_pilot.yaml"
$dataset = Join-Path $root "prepared_data\v3_retention_alignment_v4_3m_pilot_ctx2048"
$trainer = Join-Path $root "scripts\train_next_model.py"
$ownCheckpoint = Join-Path $run "latest.pt"
$parentCheckpoint = Join-Path $root "runs\fdt_v3_scaled_1216_20l_517m_retention_capability_v2_scale\latest.pt"
$stopFile = Join-Path $run "STOP_REQUESTED"
$expectedParentHash = "962F9780C0A619B90AF861BF536D29B000AC15153EDF8392AAD6A0E15CF60AFD"
$expectedDatasetManifestHash = "F068E80690E34D1D066104BAAA78E86C1A011A2E0E1CA01F0D0454C618ED1F0E"

$run = [System.IO.Path]::GetFullPath($run)
New-Item -ItemType Directory -Force -Path $run | Out-Null
foreach ($path in @($run, $config, $dataset, $trainer, $parentCheckpoint)) {
    $full = [System.IO.Path]::GetFullPath($path)
    if (-not $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside the C-only workspace: $full"
    }
    if ([System.IO.Path]::GetPathRoot($full) -ne "C:\") {
        throw "Refusing non-C storage path: $full"
    }
}
foreach ($path in @($config, $dataset, $trainer, $parentCheckpoint)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required input: $path"
    }
}
if (Test-Path -LiteralPath $stopFile) {
    throw "STOP_REQUESTED still exists: $stopFile"
}
$actualParentHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $parentCheckpoint).Hash
if ($actualParentHash -ne $expectedParentHash) {
    throw "Accepted 517M parent hash mismatch: $actualParentHash"
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

$checkpoint = if (Test-Path -LiteralPath $ownCheckpoint) { $ownCheckpoint } else { $parentCheckpoint }
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
    stage = "retention_alignment_v4_3m_pilot"
    resumed_from = $checkpoint
    parent_sha256 = $actualParentHash
    stdout = $stdout
    stderr = $stderr
    config = $config
    dataset = $dataset
    dataset_manifest_sha256 = $actualDatasetManifestHash
    graceful_stop_file = $stopFile
    storage_drive = "C:"
    target_optimizer_steps = 182
    target_tokens = 519962624
    learning_rate = 0.0000004
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
