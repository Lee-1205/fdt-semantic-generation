$ErrorActionPreference = 'Stop'

$Root = '<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM'
$Run = Join-Path $Root 'runs\fdt_v4_1_424m_alpha024_retention_smoke_500k_20260906_r2'
$Python = '<USER_HOME>\AppData\Local\Programs\Python\Python310\python.exe'
$Trainer = Join-Path $Root 'scripts\train_fdt_v4_1_retention.py'
$Guardian = Join-Path $Root 'scripts\guard_fdt_v4_1_retention_smoke.py'
$Config = Join-Path $Root 'configs\fdt_v4_1_424m_alpha024_retention_smoke_500k_20260903.yaml'
$Parent = Join-Path $Root 'runs\fdt_v4_1_exact_pointer_pilot_20260831_r3\latest.pt'
$Preflight = Join-Path $Root 'runs\fdt_v4_1_424m_alpha024_retention_smoke_preflight_20260906_r5_row_contract\training_preflight.json'
$Incident = Join-Path $Root 'artifacts\fdt_v4_1_alpha024_retention_r1_row_manifest_incident_20260906.json'

$Expected = @{
    $Trainer = 'ACD872485F8EF27A10DFFE3C4132F75E816B350A36138CFBB10E7EDE17A573E9'
    $Guardian = '9FEA45E41D4EF81CB74907DC8F0C3B38ACB15CDF20C00D3ECFB9409C40C3ED03'
    $Config = '104B3EAE2BF22D996973ACFA528FF3D36623A403315D5E2CA9656D558E661C1F'
    $Parent = '96439F598724C34F76270D614B43097547BD9E75D39D3659F62668BA5F541BFE'
    $Preflight = '8036471EA19185F87044774BD7CDB6ACF397333AABAB392C12C927C8D52D88EB'
    $Incident = 'DB79DC210BC0A4D7BDB83B616172A882477F7C3F51EC2E71D81D17A5B8AB3137'
    (Join-Path $Root 'src\fdt_rlm\models\fdt_v4.py') = '0E881F62219210F555BBEB7E65F39BF235B634DB5B89395D9590DB79B28E4270'
    (Join-Path $Root 'src\fdt_rlm\models\causal_lm.py') = 'E2A3F879EECB286FE1A49DFC98E26D909AB91F6C8E13878193F26CCE79923CD3'
    (Join-Path $Root 'src\fdt_rlm\config.py') = 'F7607F3A1002F9FCC16A63C2431BA29CE10ED5FE69560DE3652AEF3789F0CC8E'
    (Join-Path $Root 'tokenizers\fdt_v3_bpe_24k\tokenizer.json') = '4FADE0C72E8673BA1E4319C8821593BB5E62A8C27792E2412DE3BDC52D232862'
    (Join-Path $Root 'prepared_data\fdt_v4_1_alpha024_retention_t4_100k_ctx512_20260903_r2\manifest.json') = 'E02B20C03AF8B992D91066C65A58C8720C28E4953E92BA18033FB858ED58DE7F'
    (Join-Path $Root 'prepared_data\fdt_v4_1_alpha024_retention_t4_views_20260903_r1\manifest.json') = 'A7696C69B41345AAB52D251D641B68A04F97188406E108A2F93703AE10B682DC'
    (Join-Path $Root 'prepared_data\fdt_v4_1_fixed52_dual_validation_20260903_r1\manifest.json') = 'E1A85CEF360E02512B045515FE111B540A8D4DCB703CA95CA97901F0E9A4BD36'
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

$PreflightReport = Get-Content -LiteralPath $Preflight -Raw | ConvertFrom-Json
if ($PreflightReport.status -ne 'PASS' -or $PreflightReport.training_launched) {
    throw 'No-training preflight did not pass cleanly'
}
if ([int64]$PreflightReport.parameter_count -ne 426964440) { throw 'Parameter count changed' }
if ($PreflightReport.config_sha256 -ne $Expected[$Config]) { throw 'Preflight config hash changed' }
if ($PreflightReport.parent_checkpoint_sha256 -ne $Expected[$Parent]) { throw 'Preflight parent hash changed' }
if ([int]$PreflightReport.data_preflight.sources.natural.declared_rows -ne 20000) {
    throw 'Runtime natural-row contract was not preflighted'
}
if ([int]$PreflightReport.data_preflight.sources.fixed52_validation.declared_rows -ne 52) {
    throw 'Runtime fixed52-row contract was not preflighted'
}

if (Test-Path -LiteralPath $Run) {
    if ((Get-ChildItem -LiteralPath $Run -Force | Measure-Object).Count -gt 0) {
        throw "Fresh output path is already non-empty: $Run"
    }
} else {
    New-Item -ItemType Directory -Path $Run | Out-Null
}
$FreeGiB = (Get-PSDrive C).Free / 1GB
if ($FreeGiB -lt 52.0) { throw "C: is below the 52 GiB launch floor: $FreeGiB" }
$Os = Get-CimInstance Win32_OperatingSystem
if (([double]$Os.FreeVirtualMemory * 1KB / 1GB) -lt 4.0) { throw 'Free commit headroom is below 4 GiB' }
$Existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -and $_.CommandLine -match 'FDT_RLM' -and $_.CommandLine -match '(train_fdt|evaluate|diagnose)'
}
if ($Existing) { throw 'Another FDT trainer or evaluator is active' }
$GpuEvents = @(Get-WinEvent -FilterHashtable @{LogName='System';StartTime=(Get-Date).AddHours(-12)} -ErrorAction SilentlyContinue | Where-Object {
    $_.ProviderName -match 'nvlddmkm|Display' -and $_.Id -in 153,4101
})
if ($GpuEvents.Count -gt 0) { throw 'Recent GPU driver event blocks launch' }

$Stdout = Join-Path $Run 'stdout.log'
$Stderr = Join-Path $Run 'stderr.log'
$env:CUDA_MODULE_LOADING = 'LAZY'
$Process = Start-Process -FilePath $Python -ArgumentList @($Trainer, '--config', $Config, '--output-dir', $Run, '--device', 'cuda', '--allow-gpu') `
    -WorkingDirectory $Root -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath (Join-Path $Run 'train.pid') -Encoding ascii
$Active = [ordered]@{
    pid=$Process.Id; stdout=$Stdout; stderr=$Stderr; training_log=(Join-Path $Run 'training_log.jsonl')
    config=$Config; config_sha256=$Expected[$Config]; trainer=$Trainer; trainer_sha256=$Expected[$Trainer]
    guardian=$Guardian; guardian_sha256=$Expected[$Guardian]; parent=$Parent; parent_sha256=$Expected[$Parent]
    preflight=$Preflight; preflight_sha256=$Expected[$Preflight]; incident=$Incident
    storage_floor_gib=40.0; launch_floor_gib=52.0; launched_at=(Get-Date -Format o)
}
$Active | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Run 'active_logs.json') -Encoding utf8

$TrainingLog = Join-Path $Run 'training_log.jsonl'
$Deadline = (Get-Date).AddMinutes(15)
while (-not (Test-Path -LiteralPath $TrainingLog)) {
    $Process.Refresh()
    if ($Process.HasExited) { throw "Trainer exited before creating its log: $($Process.ExitCode)" }
    if ((Get-Date) -ge $Deadline) { throw 'Trainer did not create its log within 15 minutes' }
    Start-Sleep -Seconds 2
}

$GuardianStdout = Join-Path $Run 'guardian_stdout.log'
$GuardianStderr = Join-Path $Run 'guardian_stderr.log'
$GuardianProcess = Start-Process -FilePath $Python -ArgumentList @($Guardian, '--run-dir', $Run, '--poll-seconds', '15') `
    -WorkingDirectory $Root -RedirectStandardOutput $GuardianStdout -RedirectStandardError $GuardianStderr -WindowStyle Hidden -PassThru
$GuardianProcess.Id | Set-Content -LiteralPath (Join-Path $Run 'guardian.pid') -Encoding ascii
$Active.guardian_pid=$GuardianProcess.Id; $Active.guardian_stdout=$GuardianStdout; $Active.guardian_stderr=$GuardianStderr
$Active | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Run 'active_logs.json') -Encoding utf8
[pscustomobject]@{run=$Run;trainer_pid=$Process.Id;guardian_pid=$GuardianProcess.Id;launch_free_gib=[math]::Round($FreeGiB,3)} | ConvertTo-Json
