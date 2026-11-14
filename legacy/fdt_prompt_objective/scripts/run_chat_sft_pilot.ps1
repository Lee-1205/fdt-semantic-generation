param(
    [string]$Python = "python"
)

# Hugging Face writes benign warnings to stderr. Windows PowerShell promotes
# those lines to NativeCommandError when Stop is used, even with redirection.
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DataDir = "prepared_data\chat_sft_pilot_5m"
$RunDir = "runs\fdt_chat_sft_pilot_5m"
$ArtifactDir = "artifacts\chat_sft_pilot_5m"
New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null

if (-not (Test-Path "$DataDir\manifest.json")) {
    & $Python scripts\prepare_chat_sft.py `
        --output-dir $DataDir `
        --source-profile pilot_quality `
        --target-assistant-tokens 5000000 `
        --validation-examples 1000 `
        --max-length 512 `
        --max-answer-tokens 384 `
        *> "$ArtifactDir\prepare.log"
    if ($LASTEXITCODE -ne 0) { throw "Chat SFT data preparation failed with exit code $LASTEXITCODE" }
}

& $Python scripts\train_chat_sft.py `
    --checkpoint runs\fdt_broad_mix_1b\latest.pt `
    --dataset-dir $DataDir `
    --output-dir $RunDir `
    --batch-size 8 `
    --epochs 1 `
    --lr 0.000002 `
    --prompt-lm-weight 0.05 `
    --broad-validation-rows 256 `
    --max-broad-loss-increase 0.02 `
    --min-chat-improvement 0.05 `
    --eval-every 250 `
    --save-every 500 `
    --freeze-routing `
    *> "$ArtifactDir\train.log"
$TrainExit = $LASTEXITCODE

$Candidate = "$RunDir\best.pt"
if (-not (Test-Path $Candidate)) { $Candidate = "$RunDir\latest.pt" }
if (Test-Path $Candidate) {
    & $Python scripts\generate_probe.py `
        --model_type fdt_optimized `
        --checkpoint $Candidate `
        --prompt_file eval_prompts\base_probe_prompts.jsonl `
        --max_new_tokens 48 `
        --temperature 0.7 `
        --top_k 40 `
        --top_p 0.9 `
        --repetition_penalty 1.15 `
        --cache_policy on `
        --precision bf16 `
        --device cuda `
        --output_jsonl "$ArtifactDir\generation.jsonl" `
        *> "$ArtifactDir\generation.log"
    if ($LASTEXITCODE -ne 0) { throw "Generation probe failed with exit code $LASTEXITCODE" }

    & $Python scripts\check_language_gate.py `
        --checkpoint $Candidate `
        --model-type fdt_optimized `
        --device cuda `
        --no-allow-accuracy-transition `
        --output "$ArtifactDir\language_gate.json" `
        *> "$ArtifactDir\language_gate.log"
    $GateExit = $LASTEXITCODE
} else {
    $GateExit = 3
}

@{
    completed_at = (Get-Date).ToString("o")
    train_exit = $TrainExit
    gate_exit = $GateExit
    candidate = $Candidate
} | ConvertTo-Json | Set-Content -Encoding utf8 "$ArtifactDir\complete.json"

exit 0
