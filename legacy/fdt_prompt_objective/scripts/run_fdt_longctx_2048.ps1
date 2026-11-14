param(
    [int]$IdleUtilization = 20,
    [int]$IdleChecks = 3,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$RunDir = Join-Path $Root "runs\fdt_v3_longctx_2048_70m"
$Latest = Join-Path $RunDir "latest.pt"
$Original = Join-Path $Root "runs\fdt_v3_language_50m\latest.pt"
$Log = Join-Path $RunDir "launcher.log"
$BenchDir = Join-Path $Root "artifacts\inference_50m_final"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
New-Item -ItemType Directory -Force -Path $BenchDir | Out-Null

$idle = 0
while ($idle -lt $IdleChecks) {
    $raw = & nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
    $utilization = [int](($raw | Select-Object -First 1).Trim())
    if ($utilization -le $IdleUtilization) {
        $idle += 1
    } else {
        $idle = 0
    }
    if ($idle -lt $IdleChecks) {
        Start-Sleep -Seconds $PollSeconds
    }
}

$benchmarks = @(
    @{ Model = "fdt_v3"; Checkpoint = "runs/fdt_v3_language_50m/latest.pt"; Context = 64; Output = "artifacts/inference_50m_final/fdt_ctx64.json" },
    @{ Model = "transformer"; Checkpoint = "runs/transformer_v3_language_50m/latest.pt"; Context = 64; Output = "artifacts/inference_50m_final/transformer_ctx64.json" },
    @{ Model = "fdt_v3"; Checkpoint = "runs/fdt_v3_language_50m/latest.pt"; Context = 448; Output = "artifacts/inference_50m_final/fdt_ctx448.json" },
    @{ Model = "transformer"; Checkpoint = "runs/transformer_v3_language_50m/latest.pt"; Context = 448; Output = "artifacts/inference_50m_final/transformer_ctx448.json" }
)
foreach ($case in $benchmarks) {
    & python scripts/benchmark_inference.py `
        --model_type $case.Model `
        --checkpoint $case.Checkpoint `
        --prompt_len $case.Context `
        --generate_len 64 `
        --batch_size 1 `
        --warmup_steps 4 `
        --use_cache true `
        --precision bf16 `
        --output $case.Output `
        2>&1 | Add-Content -Path $Log
    if ($LASTEXITCODE -ne 0) {
        throw "Inference benchmark failed for $($case.Model) at context $($case.Context)"
    }
}

$Resume = if (Test-Path $Latest) { $Latest } else { $Original }
$stamp = Get-Date -Format o
"[$stamp] Starting FDT 2048 continuation from $Resume" | Add-Content -Path $Log
& python scripts/train_next_model.py `
    --config configs/fdt_v3_longctx_2048_70m.yaml `
    --resume_from $Resume `
    --live_log_path runs/fdt_v3_longctx_2048_70m/training_log.jsonl `
    2>&1 | Tee-Object -FilePath $Log -Append

$exitCode = $LASTEXITCODE
"[$(Get-Date -Format o)] Training exited with code $exitCode" | Add-Content -Path $Log
exit $exitCode
