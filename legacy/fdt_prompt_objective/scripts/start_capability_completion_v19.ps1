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
$Preparer = Join-Path $Root "scripts\prepare_capability_completion_v19.py"
$PreparerHash = "524EC7CB00214BC57CADB33B9578D270A1A51890C3666D30A32065AF26444189"
$Trainer = Join-Path $Root "scripts\train_capability_completion_v19.py"
$TrainerHash = "8B8FE630C8150354C5C8CDD5BC4CEADDFF6A5D2DAA8EEC9756D078BDE4DE440A"
$Tests = Join-Path $Root "tests\test_capability_completion_v19.py"
$TestsHash = "E67CACC82B7917E9CAFBC8AB988533A7DD014AD00BBFB1B65A5DBB355673C792"
$Config = Join-Path $Root "configs\fdt_v3_capability_completion_v19_copy_grounding.yaml"
$ConfigHash = "1EE623F93FAA170E08B5DB90AD6A54DE48AA778038EA37EA1DE0188811BCDF26"
$OutputDir = Join-Path $Root "runs\fdt_v3_capability_completion_v19_copy_grounding_20260815"

foreach ($Path in @($Root, $Checkpoint, $V17Reference, $Dataset, $Preparer, $Trainer, $Tests, $Config, $OutputDir)) {
    if (-not ([System.IO.Path]::GetFullPath($Path).StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "V19 permits only C: paths: $Path"
    }
}
if ((Get-PSDrive C).Free -lt 74GB) { throw "V19 launch requires at least 74 GiB transient runtime headroom" }
$Pinned = @{
    $Checkpoint = $CheckpointHash
    $V17Reference = $V17ReferenceHash
    (Join-Path $Dataset "manifest.json") = $ManifestHash
    (Join-Path $Dataset "shards\train\shard_000000.pt") = $TrainShardHash
    (Join-Path $Dataset "shards\validation\shard_000000.pt") = $ValidationShardHash
    $Preparer = $PreparerHash
    $Trainer = $TrainerHash
    $Tests = $TestsHash
    $Config = $ConfigHash
}
foreach ($Entry in $Pinned.GetEnumerator()) {
    $Actual = (Get-FileHash -LiteralPath $Entry.Key -Algorithm SHA256).Hash
    if ($Actual -ne $Entry.Value) { throw "Pinned V19 hash mismatch: $($Entry.Key)" }
}
$Manifest = Get-Content -LiteralPath (Join-Path $Dataset "manifest.json") -Raw | ConvertFrom-Json
if ($Manifest.post_build_exact_overlap -ne 0 -or $Manifest.source_revisions.hotpot_qa -ne "1908d6afbbead072334abe2965f91bd2709910ab") {
    throw "V19 dataset overlap or HotpotQA revision invariant failed"
}
if ($Manifest.source_provenance.factual_qa.license -ne "cc-by-sa-4.0") {
    throw "V19 factual source license provenance is missing"
}
$ActiveTrainer = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'train_capability_completion|train_scaled'
}
if ($ActiveTrainer) { throw "another FDT trainer is active: $($ActiveTrainer.ProcessId -join ',')" }
if (Test-Path -LiteralPath $OutputDir) {
    if (Get-ChildItem -LiteralPath $OutputDir -Force) { throw "Dedicated V19 output is not empty" }
} else {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $OutputDir "train.$Stamp.stdout.log"
$Stderr = Join-Path $OutputDir "train.$Stamp.stderr.log"
$Arguments = @(
    "-u", "scripts\train_capability_completion_v19.py",
    "--checkpoint", $Checkpoint,
    "--expected-checkpoint-sha256", $CheckpointHash,
    "--v17-reference-checkpoint", $V17Reference,
    "--expected-v17-reference-sha256", $V17ReferenceHash,
    "--dataset-dir", $Dataset,
    "--expected-manifest-sha256", $ManifestHash,
    "--output-dir", $OutputDir,
    "--batch-size", "10",
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
    stage = "v19_copy_grounding"
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
    preparer_sha256 = $PreparerHash
    trainer_sha256 = $TrainerHash
    tests_sha256 = $TestsHash
    config_sha256 = $ConfigHash
    factual_source = "hotpotqa/hotpot_qa"
    factual_source_revision = "1908d6afbbead072334abe2965f91bd2709910ab"
    factual_source_license = "cc-by-sa-4.0"
    output_dir = $OutputDir
    transient_storage_floor_gib = 74
    post_completion_storage_target_gib = 80
    training = $true
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDir "active_logs.json") -Encoding utf8

Write-Output "Started V19 copy-and-grounding training PID $($Process.Id)"
