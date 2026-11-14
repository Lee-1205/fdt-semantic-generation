$ErrorActionPreference = 'Stop'

$Root = '<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM'
$Run = Join-Path $Root 'runs\fdt_v4_1_424m_rope_stage_b_micro_20260902_r1'
$Python = '<USER_HOME>\AppData\Local\Programs\Python\Python310\python.exe'
$Trainer = Join-Path $Root 'scripts\train_fdt_v4_curriculum_bridge.py'
$Config = Join-Path $Root 'configs\fdt_v4_1_424m_rope_stage_b_micro_1p87m_20260902.yaml'
$Parent = Join-Path $Root 'runs\fdt_v4_1_exact_pointer_pilot_20260831_r3\latest.pt'
$NovelManifest = Join-Path $Root 'prepared_data\fdt_v4_1_stage_b_novel_t3_20260901\manifest.json'
$NaturalManifest = Join-Path $Root 'prepared_data\fdt_v4_1_stage_b_novel_t3_20260901\natural\manifest.json'
$FactualManifest = Join-Path $Root 'prepared_data\fdt_v4_1_stage_b_novel_t3_20260901\factual\manifest.json'
$StaticProbe = Join-Path $Root 'artifacts\fdt_v4_1_stage_b_static_alpha_probe_20260901.json'
$FineProbe = Join-Path $Root 'artifacts\fdt_v4_1_stage_b_static_alpha_fine_probe_20260901.json'
$Preflight = Join-Path $Root 'runs\fdt_v4_1_424m_rope_stage_b_micro_preflight_20260901_r2\training_preflight.json'

$Expected = @{
    $Trainer = '434E36562C5A7483250D8396764C36215AABFB5F8BEB96937EDB26A42224E12C'
    (Join-Path $Root 'src\fdt_rlm\models\fdt_v4.py') = '0E881F62219210F555BBEB7E65F39BF235B634DB5B89395D9590DB79B28E4270'
    (Join-Path $Root 'src\fdt_rlm\models\causal_lm.py') = 'E2A3F879EECB286FE1A49DFC98E26D909AB91F6C8E13878193F26CCE79923CD3'
    (Join-Path $Root 'src\fdt_rlm\config.py') = 'F7607F3A1002F9FCC16A63C2431BA29CE10ED5FE69560DE3652AEF3789F0CC8E'
    $Config = 'F07876442F06EA49A0392543317B606F54F85F461CA05B1FFA381D7A67D103C9'
    $Parent = '96439F598724C34F76270D614B43097547BD9E75D39D3659F62668BA5F541BFE'
    $NovelManifest = '0B1BD976AF12CBC75FC38AAD93C6FCF030430B218BBFAABBE0B3BB42A13B453B'
    $NaturalManifest = '87C9225D2DBAD1237EBB364C7783C9BE01705414859E2D28DADB3AE05B786EA5'
    $FactualManifest = '6C195C2306F90A129D22C2849FA2F7CF55F01401F1247023AEAF2E493ADE899C'
    $StaticProbe = '8D6CDED0AB9CA98E54E86E26C7AA63C47136352691401B8EE539DE27D8A085D3'
    $FineProbe = 'E16FE602549590D63C9B1F8DA3DE5189FB552A28A9FB571200495F3E3ABEB8FD'
    $Preflight = '17B707E44660E914FD5F960F2CDFF36A1C6ECF0F83BD2323CE71F20D5ECEB66A'
}
foreach ($Entry in $Expected.GetEnumerator()) {
    $Full = [System.IO.Path]::GetFullPath($Entry.Key)
    if (-not $Full.StartsWith($Root + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "C-only workspace contract violated: $Full"
    }
    if (-not (Test-Path -LiteralPath $Full)) { throw "Pinned input missing: $Full" }
    if ((Get-FileHash -LiteralPath $Full -Algorithm SHA256).Hash -ne $Entry.Value) {
        throw "Pinned SHA-256 mismatch: $Full"
    }
}

$Novel = Get-Content -LiteralPath $NovelManifest -Raw | ConvertFrom-Json
if ($Novel.exact_overlap_with_stage_a -ne 0 -or $Novel.local_exact_duplicates -ne 0) {
    throw 'Stage B data is not globally exact-novel against Stage A'
}
$Probe = Get-Content -LiteralPath $FineProbe -Raw | ConvertFrom-Json
$ParentLoss = [double]$Probe.parent_validation_loss
$Alpha030 = $Probe.candidate | Where-Object { [math]::Abs([double]$_.alpha - 0.30) -lt 1e-12 }
$Alpha035 = $Probe.candidate | Where-Object { [math]::Abs([double]$_.alpha - 0.35) -lt 1e-12 }
if ($null -eq $Alpha030 -or ([double]$Alpha030.validation_loss / $ParentLoss - 1.0) -gt 0.01) {
    throw 'Static alpha 0.30 did not pass the 1% V20 loss gate'
}
if ($null -eq $Alpha035 -or ([double]$Alpha035.validation_loss / $ParentLoss - 1.0) -le 0.01) {
    throw 'Static alpha boundary evidence changed unexpectedly'
}
$PreflightReport = Get-Content -LiteralPath $Preflight -Raw | ConvertFrom-Json
if ($PreflightReport.status -ne 'PASS' -or $PreflightReport.training_launched) {
    throw 'No-training Stage B preflight did not pass cleanly'
}

if (Test-Path -LiteralPath $Run) {
    if ((Get-ChildItem -LiteralPath $Run -Force | Measure-Object).Count -gt 0) {
        throw "Fresh output path is already non-empty: $Run"
    }
} else {
    New-Item -ItemType Directory -Path $Run | Out-Null
}
$FreeGiB = (Get-PSDrive C).Free / 1GB
if ($FreeGiB -lt 13.0) { throw "C: free space is below the 13 GiB launch floor: $FreeGiB" }
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
    alpha_probe = $FineProbe
    alpha_probe_sha256 = $Expected[$FineProbe]
    launched_at = (Get-Date -Format o)
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Run 'active_logs.json') -Encoding utf8
$Process.Id
