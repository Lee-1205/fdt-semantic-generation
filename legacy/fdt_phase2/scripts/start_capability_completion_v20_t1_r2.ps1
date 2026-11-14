param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Parent = Join-Path $Root "runs\fdt_v3_capability_completion_v18_calibrated_retention_20260815_r4_resume\latest.pt"
$ParentHash = "23C5539B7EF870311DB36E65F21896343A8F53D5203F603E83DD2F5CD02745B6"
$Reference = Join-Path $Root "runs\fdt_v3_capability_completion_v17_balanced_20260814\latest.pt"
$ReferenceHash = "A7C418B6014412E39D6C9C659C7614A0677D7D062CE1A6A7239A7DD8CF0A8D92"
$Dataset = Join-Path $Root "prepared_data\v3_capability_completion_v20_balanced_scale_t1_22k5_ctx512"
$ManifestHash = "B8A25CFB1A62BC14455C85F2A861DA1FA566292EE7AC35B8EED6617D424DE781"
$TrainHash = "5A1DE99F5110EFA545A53CF9FD3CC0473AF153E1FA2F9DA777DF142221805D00"
$ValidationHash = "E044E8B8D2382825994A1115FC95A5553ADF556DE503CB4DBA83BFC6CDC1FFE6"
$Preparer = Join-Path $Root "scripts\prepare_capability_completion_v20.py"
$PreparerHash = "DDD0A05E6828EE63645D29BC4EF5DE770CF7ED26A20C39B915E6B0FF53F24FBE"
$Trainer = Join-Path $Root "scripts\train_capability_completion_v20.py"
$TrainerHash = "914BE0D7BA17D2830C29BAC33ECE4F249D68AD94D437790D11C43DA4E81A8815"
$TestA = Join-Path $Root "tests\test_capability_completion_v20.py"
$TestAHash = "43F7EBEFB6816E97E7336F24AFF193788B236DB8B03699C8BD8EAA9B4111D435"
$TestB = Join-Path $Root "tests\test_capability_completion_v20_data.py"
$TestBHash = "F99AC9435C913262FC4283CFF4B159439A7292A8C45D5A5889047155E768038D"
$Config = Join-Path $Root "configs\fdt_v3_capability_completion_v20_balanced_scale_t1_r2.yaml"
$ConfigHash = "1D181E0BE378E3E6319CB1DBE05C9A18F1E50FE7748A3193006CF95C44FFEC3B"
$OutputDir = Join-Path $Root "runs\fdt_v3_capability_completion_v20_balanced_scale_t1_20260816_r2"
$LaunchArtifact = Join-Path $Root "artifacts\fdt_v20_balanced_scale_t1_r2_launch_20260816.json"

$PinnedFiles = [ordered]@{
    $Parent = $ParentHash
    $Reference = $ReferenceHash
    (Join-Path $Dataset "manifest.json") = $ManifestHash
    (Join-Path $Dataset "shards\train\shard_000000.pt") = $TrainHash
    (Join-Path $Dataset "shards\validation\shard_000000.pt") = $ValidationHash
    $Preparer = $PreparerHash
    $Trainer = $TrainerHash
    $TestA = $TestAHash
    $TestB = $TestBHash
    $Config = $ConfigHash
}
foreach ($Path in @($Root, $OutputDir, $LaunchArtifact) + @($PinnedFiles.Keys)) {
    $FullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $FullPath.StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "V20 permits only C: paths: $FullPath"
    }
}
foreach ($Item in $PinnedFiles.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $Item.Key)) { throw "Missing pinned V20 input: $($Item.Key)" }
    if ((Get-FileHash -LiteralPath $Item.Key -Algorithm SHA256).Hash -ne $Item.Value) {
        throw "V20 pinned hash mismatch: $($Item.Key)"
    }
}
if (Test-Path -LiteralPath $LaunchArtifact) { throw "V20 launch artifact already exists" }
$ManifestPayload = Get-Content -LiteralPath (Join-Path $Dataset "manifest.json") -Raw | ConvertFrom-Json
if ([double]$ManifestPayload.prompt_lm_weight -ne 0.35) { throw "V20 prompt-LM cross-artifact preflight failed" }
if ([double]$ManifestPayload.eos_loss_weight -ne 2.0 -or [double]$ManifestPayload.grounding_loss_weight -ne 1.5 -or [double]$ManifestPayload.signature_loss_weight -ne 1.25) {
    throw "V20 objective cross-artifact preflight failed"
}
if (Test-Path -LiteralPath $OutputDir) {
    if (Get-ChildItem -LiteralPath $OutputDir -Force) { throw "V20 output directory is not fresh" }
} else {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}
$Disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
$FreeGiB = $Disk.FreeSpace / 1GB
if ($FreeGiB -lt 80) { throw "V20 launch requires at least 80 GiB free before training" }
$BadProcess = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '(?i)woolypooly|progpow|stratum' })
$BadTask = Get-ScheduledTask -TaskName "GoogleUpdateTaskSYSTEM" -ErrorAction SilentlyContinue
if ($BadProcess -or $BadTask) { throw "V20 security preflight failed" }

$PytestTemp = Join-Path $Root "artifacts\pytest_v20_launch_20260816"
New-Item -ItemType Directory -Force -Path $PytestTemp | Out-Null
$PreviousTemp = $env:TEMP
$PreviousTmp = $env:TMP
$env:TEMP = $PytestTemp
$env:TMP = $PytestTemp
try {
    & $Python -m pytest -q $TestA $TestB
    if ($LASTEXITCODE -ne 0) { throw "V20 focused tests failed" }
} finally {
    $env:TEMP = $PreviousTemp
    $env:TMP = $PreviousTmp
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $OutputDir "train.$Stamp.stdout.log"
$Stderr = Join-Path $OutputDir "train.$Stamp.stderr.log"
$Arguments = @(
    "scripts\train_capability_completion_v20.py",
    "--checkpoint", $Parent,
    "--expected-checkpoint-sha256", $ParentHash,
    "--v17-reference-checkpoint", $Reference,
    "--expected-v17-reference-sha256", $ReferenceHash,
    "--dataset-dir", $Dataset,
    "--expected-manifest-sha256", $ManifestHash,
    "--output-dir", $OutputDir,
    "--batch-size", "10",
    "--micro-batch-size", "5",
    "--lr", "1.25e-8",
    "--weight-decay", "0.01",
    "--prompt-lm-weight", "0.35",
    "--eos-loss-weight", "2.0",
    "--grounding-loss-weight", "1.5",
    "--signature-loss-weight", "1.25",
    "--smoke-step", "500",
    "--checkpoint-interval", "1500",
    "--seed", "20260820",
    "--device", "cuda"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $OutputDir "train.pid") -Encoding ascii
$Active = [ordered]@{
    stage = "capability_completion_v20_balanced_scale_tranche_1_r2"
    pid = $Process.Id
    started_at = (Get-Date).ToString("o")
    stdout = $Stdout
    stderr = $Stderr
    training_log = (Join-Path $OutputDir "training_log.jsonl")
    parent = $Parent
    parent_sha256 = $ParentHash
    hard_reference = $Reference
    hard_reference_sha256 = $ReferenceHash
    dataset = $Dataset
    manifest_sha256 = $ManifestHash
    train_shard_sha256 = $TrainHash
    validation_shard_sha256 = $ValidationHash
    trainer_sha256 = $TrainerHash
    preparer_sha256 = $PreparerHash
    config_sha256 = $ConfigHash
    output = $OutputDir
    rows = 22500
    active_tokens = 5159159
    target_steps = 2250
    smoke_step = 500
    periodic_recovery_step = 1500
    planned_total_tranches = 4
    planned_total_active_tokens = 20000000
    training = $true
}
$Active | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDir "active_logs.json") -Encoding utf8
$Active | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $LaunchArtifact -Encoding utf8

Write-Output "Started V20 balanced-scale tranche 1 PID $($Process.Id)"
