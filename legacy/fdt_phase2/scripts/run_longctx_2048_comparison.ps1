$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Test-RunComplete([string]$RunDirectory) {
    $checkpoint = Join-Path $RunDirectory "latest.pt"
    $log = Join-Path $RunDirectory "training_log.jsonl"
    return (Test-Path $checkpoint) -and (Test-Path $log) -and `
        (Select-String -LiteralPath $log -Pattern '"event": "complete"' -Quiet)
}

if (-not (Test-RunComplete "runs\fdt_v3_longctx_2048_70m")) {
    python scripts\train_next_model.py `
        --config configs\fdt_v3_longctx_2048_70m.yaml `
        --resume_from runs\fdt_v3_language_50m\latest.pt `
        --auto_onk
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-RunComplete "runs\transformer_v3_longctx_2048_70m")) {
    python scripts\train_next_model.py `
        --config configs\transformer_v3_longctx_2048_70m.yaml `
        --resume_from runs\transformer_v3_language_50m\latest.pt
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

exit 0
