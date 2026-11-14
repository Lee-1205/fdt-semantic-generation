$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$PilotCheckpoint = Join-Path $Root "runs\transformer_v3_language_pilot_5m\latest.pt"
$PilotLog = Join-Path $Root "runs\transformer_v3_language_pilot_5m\training_log.jsonl"
$PilotComplete = $false
if ((Test-Path $PilotCheckpoint) -and (Test-Path $PilotLog)) {
    $PilotComplete = Select-String -LiteralPath $PilotLog -Pattern '"event": "complete"' -Quiet
}

if (-not $PilotComplete) {
    python scripts\train_next_model.py --config configs\transformer_v3_language_pilot.yaml
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python scripts\train_next_model.py `
    --config configs\transformer_v3_language_50m.yaml `
    --resume_from runs\transformer_v3_language_pilot_5m\latest.pt
exit $LASTEXITCODE
