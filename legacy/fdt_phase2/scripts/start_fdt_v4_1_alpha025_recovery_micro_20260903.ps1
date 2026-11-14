$ErrorActionPreference = 'Stop'

$Root = '<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM'
$Run = Join-Path $Root 'runs\fdt_v4_1_424m_alpha025_recovery_micro_20260903_r1'
$Python = '<USER_HOME>\AppData\Local\Programs\Python\Python310\python.exe'
$Trainer = Join-Path $Root 'scripts\train_fdt_v4_curriculum_bridge.py'
$Config = Join-Path $Root 'configs\fdt_v4_1_424m_alpha025_recovery_micro_1p87m_20260902.yaml'
$Parent = Join-Path $Root 'runs\fdt_v4_1_exact_pointer_pilot_20260831_r3\latest.pt'
$NovelManifest = Join-Path $Root 'prepared_data\fdt_v4_1_stage_b_novel_t3_20260901\manifest.json'
$NaturalManifest = Join-Path $Root 'prepared_data\fdt_v4_1_stage_b_novel_t3_20260901\natural\manifest.json'
$FactualManifest = Join-Path $Root 'prepared_data\fdt_v4_1_stage_b_novel_t3_20260901\factual\manifest.json'
$DualAggregateProbe = Join-Path $Root 'artifacts\fdt_v4_1_fixed52_dual_aggregate_probe_20260901.json'
$Preflight = Join-Path $Root 'runs\fdt_v4_1_424m_alpha025_recovery_micro_preflight_20260903_r1\training_preflight.json'

$Expected = @{
    $Trainer = '434E36562C5A7483250D8396764C36215AABFB5F8BEB96937EDB26A42224E12C'
    (Join-Path $Root 'src\fdt_rlm\models\fdt_v4.py') = '0E881F62219210F555BBEB7E65F39BF235B634DB5B89395D9590DB79B28E4270'
    (Join-Path $Root 'src\fdt_rlm\models\causal_lm.py') = 'E2A3F879EECB286FE1A49DFC98E26D909AB91F6C8E13878193F26CCE79923CD3'
    (Join-Path $Root 'src\fdt_rlm\config.py') = 'F7607F3A1002F9FCC16A63C2431BA29CE10ED5FE69560DE3652AEF3789F0CC8E'
    $Config = 'BFEAA93C76492BFB46457491E3EE2FC0916B802072B7CDD7FF4C3B37B2B84D32'
    $Parent = '96439F598724C34F76270D614B43097547BD9E75D39D3659F62668BA5F541BFE'
    $NovelManifest = '0B1BD976AF12CBC75FC38AAD93C6FCF030430B218BBFAABBE0B3BB42A13B453B'
    $NaturalManifest = '87C9225D2DBAD1237EBB364C7783C9BE01705414859E2D28DADB3AE05B786EA5'
    $FactualManifest = '6C195C2306F90A129D22C2849FA2F7CF55F01401F1247023AEAF2E493ADE899C'
    $DualAggregateProbe = '9B86BBA46C46176F18E9C81B63F791361EA9FA4BB584DC4768006A81727BF43E'
    $Preflight = '49F319260C9558C1019B790948F09AEF18A1CB228CA3D2158F073CED329E83E0'
}

foreach ($Entry in $Expected.GetEnumerator()) {
    $Full = [System.IO.Path]::GetFullPath($Entry.Key)
    if (-not $Full.StartsWith($Root + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "C-only workspace contract violated: $Full"
    }
    if (-not (Test-Path -LiteralPath $Full -PathType Leaf)) { throw "Pinned input missing: $Full" }
    if ((Get-FileHash -LiteralPath $Full -Algorithm SHA256).Hash -ne $Entry.Value) {
        throw "Pinned SHA-256 mismatch: $Full"
    }
}

$Novel = Get-Content -LiteralPath $NovelManifest -Raw | ConvertFrom-Json
if ($Novel.exact_overlap_with_stage_a -ne 0 -or $Novel.local_exact_duplicates -ne 0) {
    throw 'Recovery data is not exact-novel against Stage A'
}
$Probe = Get-Content -LiteralPath $DualAggregateProbe -Raw | ConvertFrom-Json
$ParentRows = $Probe.parent_fixed_row_aggregates
$Alpha025 = $Probe.candidate | Where-Object { [math]::Abs([double]$_.alpha - 0.25) -lt 1e-12 }
if ($null -eq $Alpha025) { throw 'Pinned alpha 0.25 baseline is missing' }
if (([double]$Alpha025.token_weighted_nll / [double]$ParentRows.token_weighted_nll - 1.0) -gt 0.01) {
    throw 'Alpha 0.25 already fails the token-weighted 1% gate'
}
if (([double]$Alpha025.row_mean_nll / [double]$ParentRows.row_mean_nll - 1.0) -le 0.01) {
    throw 'The row-balanced recovery target is no longer reproducible'
}
$PreflightReport = Get-Content -LiteralPath $Preflight -Raw | ConvertFrom-Json
if ($PreflightReport.status -ne 'PASS' -or $PreflightReport.training_launched) {
    throw 'No-training preflight did not pass cleanly'
}

if (Test-Path -LiteralPath $Run) {
    if ((Get-ChildItem -LiteralPath $Run -Force | Measure-Object).Count -gt 0) {
        throw "Fresh output path is already non-empty: $Run"
    }
} else {
    New-Item -ItemType Directory -Path $Run | Out-Null
}
if ((Get-PSDrive C).Free / 1GB -lt 48.0) {
    throw 'C: free space is below the 48 GiB launch floor'
}
$Existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'train_fdt_v4_curriculum_bridge.py'
}
if ($Existing) { throw 'Another FDT v4 bridge trainer is already running' }
$GpuEvents = @(Get-WinEvent -FilterHashtable @{ LogName='System'; StartTime=(Get-Date).AddHours(-12) } -ErrorAction SilentlyContinue | Where-Object {
    $_.ProviderName -match 'nvlddmkm|Display' -and $_.Id -in 153, 4101
})
if ($GpuEvents.Count -gt 0) { throw 'Recent GPU driver reset/display event blocks launch' }

$Stdout = Join-Path $Run 'stdout.log'
$Stderr = Join-Path $Run 'stderr.log'
$env:CUDA_MODULE_LOADING = 'LAZY'
$Arguments = @($Trainer, '--config', $Config, '--output-dir', $Run, '--device', 'cuda', '--allow-gpu')
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr `
    -WindowStyle Hidden -PassThru
Set-Content -LiteralPath (Join-Path $Run 'train.pid') -Value $Process.Id -Encoding ascii
@{
    pid = $Process.Id
    stdout = $Stdout
    stderr = $Stderr
    training_log = (Join-Path $Run 'training_log.jsonl')
    config = $Config
    config_sha256 = $Expected[$Config]
    trainer_sha256 = $Expected[$Trainer]
    parent = $Parent
    parent_sha256 = $Expected[$Parent]
    data_manifest = $NovelManifest
    data_manifest_sha256 = $Expected[$NovelManifest]
    fixed52_probe = $DualAggregateProbe
    fixed52_probe_sha256 = $Expected[$DualAggregateProbe]
    storage_floor_gib = 40.0
    launched_at = (Get-Date -Format o)
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Run 'active_logs.json') -Encoding utf8
$Process.Id
