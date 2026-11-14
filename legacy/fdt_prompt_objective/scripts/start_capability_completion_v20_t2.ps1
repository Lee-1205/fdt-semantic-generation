param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Parent = Join-Path $Root "runs\fdt_v3_capability_completion_v20_balanced_scale_t1_20260816_r2\latest.pt"
$ParentHash = "7EE3F88D319928DD2D3F2542290F55FFCD036DCBB32A8AB22437C511E5890179"
$Reference = Join-Path $Root "runs\fdt_v3_capability_completion_v17_balanced_20260814\latest.pt"
$ReferenceHash = "A7C418B6014412E39D6C9C659C7614A0677D7D062CE1A6A7239A7DD8CF0A8D92"
$Dataset = Join-Path $Root "prepared_data\v3_capability_completion_v20_balanced_scale_t2_22k5_ctx512"
$ManifestHash = "1308D5A91123B92BF252197CCEAA8CC540869ABE455F3A0DA2F59FBAEBA2C470"
$TrainHash = "77C5C67D033C614DB32EC07BD79486A5AF5EDD00118AE8F1B4091CFFB8F21ED5"
$ValidationHash = "E044E8B8D2382825994A1115FC95A5553ADF556DE503CB4DBA83BFC6CDC1FFE6"
$Preparer = Join-Path $Root "scripts\prepare_capability_completion_v20_t2.py"
$PreparerHash = "13FF9B2724C9FAAB426D2B70F08F013D9CF2B8BCB443EFF5F9BDB7ECD763E080"
$Trainer = Join-Path $Root "scripts\train_capability_completion_v20_t2.py"
$TrainerHash = "1BF1AD0F23B3FFF466C713D460177A108743B4CB98B5D916265DC1E6D50C1EA1"
$TestA = Join-Path $Root "tests\test_capability_completion_v20_t2.py"
$TestAHash = "13BDCC02B775C1350B07B5CAAF6902F333BE3105BB02948DF4AFCA66B8CABAA3"
$TestB = Join-Path $Root "tests\test_capability_completion_v20_t2_data.py"
$TestBHash = "A06B7E05BF16F46F31A478BAA814C900A426BDB5B8103768CCE52698A4A3A836"
$Config = Join-Path $Root "configs\fdt_v3_capability_completion_v20_balanced_scale_t2.yaml"
$ConfigHash = "E3D52671FCEE31BDF9D6F476A3009A47491D586754BB2B489EF2DEFC10E56B77"
$OutputDir = Join-Path $Root "runs\fdt_v3_capability_completion_v20_balanced_scale_t2_20260816"
$LaunchArtifact = Join-Path $Root "artifacts\fdt_v20_balanced_scale_t2_launch_20260816.json"

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
        throw "V20 tranche 2 permits only C: paths: $FullPath"
    }
}
foreach ($Item in $PinnedFiles.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $Item.Key)) { throw "Missing pinned V20 tranche-2 input: $($Item.Key)" }
    if ((Get-FileHash -LiteralPath $Item.Key -Algorithm SHA256).Hash -ne $Item.Value) {
        throw "V20 tranche-2 pinned hash mismatch: $($Item.Key)"
    }
}
if (Test-Path -LiteralPath $LaunchArtifact) { throw "V20 tranche-2 launch artifact already exists" }
$ManifestPayload = Get-Content -LiteralPath (Join-Path $Dataset "manifest.json") -Raw | ConvertFrom-Json
if ([double]$ManifestPayload.prompt_lm_weight -ne 0.35 -or [double]$ManifestPayload.eos_loss_weight -ne 2.0 -or [double]$ManifestPayload.signature_loss_weight -ne 1.25) {
    throw "V20 tranche-2 objective cross-artifact preflight failed"
}
$ExpectedCategory = @{ factual_qa = 0.15; retrieval = 0.25; python_code = 0.15; json = 0.20; natural_lm = 0.25 }
foreach ($Name in $ExpectedCategory.Keys) {
    if ([double]$ManifestPayload.category_objective_weights.$Name -ne [double]$ExpectedCategory[$Name]) {
        throw "V20 tranche-2 category weight mismatch: $Name"
    }
}
$ExpectedGrounding = @{ factual_qa = 1.25; retrieval = 2.25; json = 2.25 }
foreach ($Name in $ExpectedGrounding.Keys) {
    if ([double]$ManifestPayload.category_grounding_loss_weights.$Name -ne [double]$ExpectedGrounding[$Name]) {
        throw "V20 tranche-2 grounding weight mismatch: $Name"
    }
}
if ([int]$ManifestPayload.post_build_exact_overlap -ne 0 -or [int]$ManifestPayload.scale_plan.tranche_index -ne 2) {
    throw "V20 tranche-2 overlap or tranche identity preflight failed"
}
if (Test-Path -LiteralPath $OutputDir) {
    if (Get-ChildItem -LiteralPath $OutputDir -Force) { throw "V20 tranche-2 output directory is not fresh" }
} else {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}
$Disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
$FreeGiB = $Disk.FreeSpace / 1GB
if ($FreeGiB -lt 80) { throw "V20 tranche-2 launch requires at least 80 GiB free before training" }
$BadProcess = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '(?i)woolypooly|progpow|stratum' })
$BadTask = Get-ScheduledTask -TaskName "GoogleUpdateTaskSYSTEM" -ErrorAction SilentlyContinue
if ($BadProcess -or $BadTask) { throw "V20 tranche-2 security preflight failed" }

$PytestTemp = Join-Path $Root "artifacts\pytest_v20_t2_launch_20260816"
New-Item -ItemType Directory -Force -Path $PytestTemp | Out-Null
$PreviousTemp = $env:TEMP
$PreviousTmp = $env:TMP
$env:TEMP = $PytestTemp
$env:TMP = $PytestTemp
try {
    & $Python -m pytest -q $TestA $TestB
    if ($LASTEXITCODE -ne 0) { throw "V20 tranche-2 focused tests failed" }
} finally {
    $env:TEMP = $PreviousTemp
    $env:TMP = $PreviousTmp
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $OutputDir "train.$Stamp.stdout.log"
$Stderr = Join-Path $OutputDir "train.$Stamp.stderr.log"
$Arguments = @(
    "scripts\train_capability_completion_v20_t2.py",
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
    "--factual-grounding-loss-weight", "1.25",
    "--retrieval-grounding-loss-weight", "2.25",
    "--json-grounding-loss-weight", "2.25",
    "--signature-loss-weight", "1.25",
    "--smoke-step", "500",
    "--checkpoint-interval", "3000",
    "--seed", "20260821",
    "--device", "cuda"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $OutputDir "train.pid") -Encoding ascii
$Active = [ordered]@{
    stage = "capability_completion_v20_balanced_scale_tranche_2"
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
    launcher_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash
    output = $OutputDir
    rows = 22500
    active_tokens = 5287872
    target_steps = 2250
    smoke_step = 500
    periodic_recovery_step = $null
    planned_total_tranches = 4
    completed_tranches_before_stage = 1
    planned_total_active_tokens = 20000000
    training = $true
}
$Active | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutputDir "active_logs.json") -Encoding utf8
$Active | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $LaunchArtifact -Encoding utf8

Write-Output "Started V20 balanced-scale tranche 2 PID $($Process.Id)"
