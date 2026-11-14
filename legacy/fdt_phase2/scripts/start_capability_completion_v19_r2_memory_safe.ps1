param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Checkpoint = Join-Path $Root "runs\fdt_v3_capability_completion_v18_calibrated_retention_20260815_r4_resume\latest.pt"
$CheckpointHash = "23C5539B7EF870311DB36E65F21896343A8F53D5203F603E83DD2F5CD02745B6"
$V17Reference = Join-Path $Root "runs\fdt_v3_capability_completion_v17_balanced_20260814\latest.pt"
$V17ReferenceHash = "A7C418B6014412E39D6C9C659C7614A0677D7D062CE1A6A7239A7DD8CF0A8D92"
$Dataset = Join-Path $Root "prepared_data\v3_capability_completion_v19_copy_grounding_5k_ctx512"
$ManifestHash = "5495AAF32BD0A114F8E6426C6A79212B3A21282ABF906F96A4A33F537D69A86C"
$TrainShardHash = "539F3CE2FAC7558954A7B6FDEF32F87855EB843ACAC4D604E3844F6065D74FC5"
$ValidationShardHash = "E044E8B8D2382825994A1115FC95A5553ADF556DE503CB4DBA83BFC6CDC1FFE6"
$Trainer = Join-Path $Root "scripts\train_capability_completion_v19_r2.py"
$TrainerHash = "FF6DAE23C64291773C6781FE991896858243BF4EB712392299E81CC7DAA61B13"
$Tests = Join-Path $Root "tests\test_capability_completion_v19_r2.py"
$TestsHash = "814132C5620EDEF05E37CDA03063A42CA8A8DA117F683E81256C16D189F7571B"
$Config = Join-Path $Root "configs\fdt_v3_capability_completion_v19_copy_grounding_r2_memory_safe.yaml"
$ConfigHash = "9153B87FAA6F591404479C42B6EBCFFCEDBC4C24EFB1E871C6E825AA3893BA23"
$Diagnostic = Join-Path $Root "artifacts\fdt_v19_r2_memory_safety_diagnostic_20260815.json"
$DiagnosticHash = "CAED9BDA94EE59D4F20134988DD6418A1AE99EDADD0808DC8DE1921B07C879F0"
$Incident = Join-Path $Root "artifacts\fdt_v19_r1_oom_shutdown_incident_20260815.json"
$IncidentHash = "2E72561C5DF890DE721B38BCB6D9A81CA1B035AFC9E04881FE3DC7EAC7DBEFFC"
$OutputDir = Join-Path $Root "runs\fdt_v3_capability_completion_v19_copy_grounding_20260815_r2_memory_safe"

foreach ($Path in @($Root, $Checkpoint, $V17Reference, $Dataset, $Trainer, $Tests, $Config, $Diagnostic, $Incident, $OutputDir)) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "V19 R2 permits only C: paths: $Path"
    }
}
if ((Get-PSDrive C).Free -lt 74GB) { throw "V19 R2 launch requires at least 74 GiB transient runtime headroom" }
$Pinned = @{
    $Checkpoint = $CheckpointHash
    $V17Reference = $V17ReferenceHash
    (Join-Path $Dataset "manifest.json") = $ManifestHash
    (Join-Path $Dataset "shards\train\shard_000000.pt") = $TrainShardHash
    (Join-Path $Dataset "shards\validation\shard_000000.pt") = $ValidationShardHash
    $Trainer = $TrainerHash
    $Tests = $TestsHash
    $Config = $ConfigHash
    $Diagnostic = $DiagnosticHash
    $Incident = $IncidentHash
}
foreach ($Entry in $Pinned.GetEnumerator()) {
    if ((Get-FileHash -LiteralPath $Entry.Key -Algorithm SHA256).Hash -ne $Entry.Value) {
        throw "Pinned V19 R2 hash mismatch: $($Entry.Key)"
    }
}
$MemoryEvidence = Get-Content -LiteralPath $Diagnostic -Raw | ConvertFrom-Json
if ($MemoryEvidence.status -ne "PASS" -or -not $MemoryEvidence.all_gradients_released_before_eval) {
    throw "V19 R2 requires a passing real-checkpoint memory diagnostic"
}
if ($MemoryEvidence.optimizer_state_devices_during_eval -ne "cpu") {
    throw "V19 R2 diagnostic did not offload optimizer state to CPU"
}
$ActiveTrainer = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'train_capability_completion|train_scaled'
}
if ($ActiveTrainer) { throw "another FDT trainer is active: $($ActiveTrainer.ProcessId -join ',')" }
if (Test-Path -LiteralPath $OutputDir) {
    if (Get-ChildItem -LiteralPath $OutputDir -Force) { throw "Dedicated V19 R2 output is not empty" }
} else {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $OutputDir "train.$Stamp.stdout.log"
$Stderr = Join-Path $OutputDir "train.$Stamp.stderr.log"
$Arguments = @(
    "-u", "scripts\train_capability_completion_v19_r2.py",
    "--checkpoint", $Checkpoint,
    "--expected-checkpoint-sha256", $CheckpointHash,
    "--v17-reference-checkpoint", $V17Reference,
    "--expected-v17-reference-sha256", $V17ReferenceHash,
    "--dataset-dir", $Dataset,
    "--expected-manifest-sha256", $ManifestHash,
    "--output-dir", $OutputDir,
    "--batch-size", "10",
    "--micro-batch-size", "5",
    "--lr", "0.000000025",
    "--weight-decay", "0.01",
    "--prompt-lm-weight", "0.25",
    "--eos-loss-weight", "2.0",
    "--grounding-loss-weight", "2.0",
    "--signature-loss-weight", "3.0",
    "--smoke-step", "100",
    "--max-broad-relative-regression", "0.003",
    "--max-capability-loss-regression", "0.0",
    "--max-dead-anchor-ratio", "0.01",
    "--seed", "20260819",
    "--device", "cuda"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $OutputDir "train.pid") -Encoding ascii
@{
    stage = "v19_copy_grounding_r2_memory_safe"
    pid = $Process.Id
    started_at = (Get-Date).ToString("o")
    stdout = $Stdout
    stderr = $Stderr
    checkpoint = $Checkpoint
    checkpoint_sha256 = $CheckpointHash
    v17_reference = $V17Reference
    v17_reference_sha256 = $V17ReferenceHash
    dataset = $Dataset
    manifest_sha256 = $ManifestHash
    train_shard_sha256 = $TrainShardHash
    validation_shard_sha256 = $ValidationShardHash
    trainer_sha256 = $TrainerHash
    tests_sha256 = $TestsHash
    config_sha256 = $ConfigHash
    memory_diagnostic_sha256 = $DiagnosticHash
    incident_sha256 = $IncidentHash
    effective_batch_size = 10
    micro_batch_size = 5
    optimizer_eval_placement = "cpu"
    pre_smoke_atomic_recovery = $true
    output_dir = $OutputDir
    transient_storage_floor_gib = 74
    post_completion_storage_target_gib = 80
    training = $true
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDir "active_logs.json") -Encoding utf8

Write-Output "Started memory-safe V19 R2 training PID $($Process.Id)"
