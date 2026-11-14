param(
    [string]$PythonPath = "<USER_HOME>\AppData\Local\Programs\Python\Python310\python.exe"
)

$ErrorActionPreference = "Continue"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDir = Join-Path $root "runs\fdt_broad_mix_1b"
$log = Join-Path $runDir "standalone_stdout.log"
$statusLog = Join-Path $runDir "standalone_status.log"

Push-Location $root
try {
    & $PythonPath scripts\train_next_model.py `
        --config configs\fdt_broad_mix_1b.yaml `
        --resume_from runs\fdt_broad_mix_1b\latest.pt `
        --auto_onk *>> $log
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

Add-Content -LiteralPath $statusLog -Value ("{0:o} 1B-only stage exited with code {1}." -f (Get-Date), $exitCode)
exit $exitCode
