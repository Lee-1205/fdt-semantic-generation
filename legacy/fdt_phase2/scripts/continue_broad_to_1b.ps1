param(
    [Parameter(Mandatory = $true)]
    [int]$CurrentProcessId,
    [string]$PythonPath = "<USER_HOME>\AppData\Local\Programs\Python\Python310\python.exe",
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourceRun = Join-Path $root "runs\fdt_broad_mix_350m"
$sourceLog = Join-Path $sourceRun "training_log.jsonl"
$sourceCheckpoint = Join-Path $sourceRun "onk_ready.pt"
$targetRun = Join-Path $root "runs\fdt_broad_mix_1b"
$supervisorLog = Join-Path $targetRun "supervisor.log"
$trainingLog = Join-Path $targetRun "launcher_stdout.log"
New-Item -ItemType Directory -Path $targetRun -Force | Out-Null

function Write-SupervisorLog([string]$Message) {
    $line = "{0:o} {1}" -f (Get-Date), $Message
    Add-Content -LiteralPath $supervisorLog -Value $line -Encoding UTF8
}

function Invoke-PythonLogged([string[]]$PythonArgs) {
    # Windows PowerShell 5 converts native stderr into ErrorRecord objects.
    # Warnings must not terminate a long-running stage when the global policy is Stop.
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonPath @PythonArgs *>> $trainingLog
        return [int]$LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
    }
}

Write-SupervisorLog "Monitoring PID $CurrentProcessId for the 350M stage."
while (Get-Process -Id $CurrentProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds $PollSeconds
}

$complete = $null
if (Test-Path -LiteralPath $sourceLog) {
    $complete = Get-Content -LiteralPath $sourceLog -Tail 200 |
        ForEach-Object {
            try { $_ | ConvertFrom-Json } catch { $null }
        } |
        Where-Object { $_.event -eq "complete" -and $_.status -eq "PASS" } |
        Select-Object -Last 1
}

if ($null -eq $complete -or [int64]$complete.tokens_seen -lt 350000000) {
    Write-SupervisorLog "ABORT: the 350M stage did not finish successfully."
    exit 2
}
if (-not (Test-Path -LiteralPath $sourceCheckpoint)) {
    Write-SupervisorLog "ABORT: no promoted ONK checkpoint was found."
    exit 3
}

$oneBLatest = Join-Path $targetRun "latest.pt"
if (Test-Path -LiteralPath $oneBLatest) {
    $oneBResumeCheckpoint = "runs\fdt_broad_mix_1b\latest.pt"
    Write-SupervisorLog "Resuming the 1B stage from its latest checkpoint."
} else {
    $oneBResumeCheckpoint = "runs\fdt_broad_mix_350m\onk_ready.pt"
    Write-SupervisorLog "Starting the 1B stage from the promoted 350M ONK checkpoint."
}
Push-Location $root
try {
    $exitCode = Invoke-PythonLogged @(
        "scripts\train_next_model.py",
        "--config", "configs\fdt_broad_mix_1b.yaml",
        "--resume_from", $oneBResumeCheckpoint,
        "--auto_onk"
    )
} finally {
    Pop-Location
}
Write-SupervisorLog "1B stage exited with code $exitCode."
if ($exitCode -ne 0) {
    exit $exitCode
}

$oneBCheckpoint = Join-Path $targetRun "onk_ready.pt"
if (-not (Test-Path -LiteralPath $oneBCheckpoint)) {
    Write-SupervisorLog "ABORT: the 1B stage produced no promoted ONK checkpoint."
    exit 4
}

$qualifiedCheckpoint = "runs\fdt_broad_mix_1b\onk_ready.pt"
Write-SupervisorLog "Running the 1B base-language capability gate."
Push-Location $root
try {
    $gateExitCode = Invoke-PythonLogged @(
        "scripts\check_language_gate.py",
        "--checkpoint", $qualifiedCheckpoint,
        "--device", "cuda",
        "--output", "artifacts\language_gate_1b.json"
    )
} finally {
    Pop-Location
}

if ($gateExitCode -ne 0) {
    Write-SupervisorLog "The 1B language gate failed; continuing base pretraining to 2B."
    Push-Location $root
    try {
        $twoBExitCode = Invoke-PythonLogged @(
            "scripts\train_next_model.py",
            "--config", "configs\fdt_broad_mix_2b.yaml",
            "--resume_from", "runs\fdt_broad_mix_1b\onk_ready.pt",
            "--auto_onk"
        )
    } finally {
        Pop-Location
    }
    Write-SupervisorLog "2B stage exited with code $twoBExitCode."
    if ($twoBExitCode -ne 0) {
        exit $twoBExitCode
    }
    $twoBCheckpoint = Join-Path $root "runs\fdt_broad_mix_2b\onk_ready.pt"
    if (-not (Test-Path -LiteralPath $twoBCheckpoint)) {
        Write-SupervisorLog "ABORT: the 2B stage produced no promoted ONK checkpoint."
        exit 5
    }
    $qualifiedCheckpoint = "runs\fdt_broad_mix_2b\onk_ready.pt"
    Write-SupervisorLog "Running the 2B base-language capability gate."
    Push-Location $root
    try {
        $gateExitCode = Invoke-PythonLogged @(
            "scripts\check_language_gate.py",
            "--checkpoint", $qualifiedCheckpoint,
            "--device", "cuda",
            "--output", "artifacts\language_gate_2b.json"
        )
    } finally {
        Pop-Location
    }
}

if ($gateExitCode -ne 0) {
    Write-SupervisorLog "HOLD: base language capability is still below the required gate; chat/RLM training was not started."
    exit 6
}

Write-SupervisorLog "Language gate passed. Preparing assistant-only chat SFT data."
Push-Location $root
try {
    $prepareExitCode = Invoke-PythonLogged @(
        "scripts\prepare_chat_sft.py",
        "--output-dir", "prepared_data\chat_sft_20m",
        "--target-assistant-tokens", "20000000"
    )
} finally {
    Pop-Location
}
if ($prepareExitCode -ne 0) {
    Write-SupervisorLog "Chat SFT preparation exited with code $prepareExitCode."
    exit $prepareExitCode
}

Write-SupervisorLog "Starting assistant-only chat SFT from the language-qualified checkpoint."
Push-Location $root
try {
    $chatExitCode = Invoke-PythonLogged @(
        "scripts\train_chat_sft.py",
        "--checkpoint", $qualifiedCheckpoint,
        "--dataset-dir", "prepared_data\chat_sft_20m",
        "--output-dir", "runs\fdt_chat_language_ready",
        "--batch-size", "8",
        "--epochs", "1",
        "--lr", "0.000005",
        "--prompt-lm-weight", "0.05",
        "--device", "cuda"
    )
} finally {
    Pop-Location
}
Write-SupervisorLog "Chat SFT exited with code $chatExitCode."
if ($chatExitCode -ne 0) {
    exit $chatExitCode
}

$chatCheckpoint = Join-Path $root "runs\fdt_chat_language_ready\best.pt"
if (-not (Test-Path -LiteralPath $chatCheckpoint)) {
    Write-SupervisorLog "ABORT: chat SFT did not produce a language-safe best checkpoint."
    exit 7
}

$rlmRun = Join-Path $root "runs\fdt_rlm_language_ready_action"
Write-SupervisorLog "Starting balanced RLM action training from the chat checkpoint."
Push-Location $root
try {
    $rlmExitCode = Invoke-PythonLogged @(
        "scripts\train_rlm_actions.py",
        "--checkpoint", "runs\fdt_chat_language_ready\best.pt",
        "--output-dir", "runs\fdt_rlm_language_ready_action",
        "--tasks", "artifacts\rlm_tasks_train_10k.jsonl",
        "--max-examples", "10000",
        "--epochs", "1",
        "--batch-size", "8",
        "--lr", "0.000005",
        "--prompt-lm-weight", "0.1",
        "--validation-rows", "64",
        "--device", "cuda"
    )
} finally {
    Pop-Location
}
Write-SupervisorLog "RLM action stage exited with code $rlmExitCode."
if ($rlmExitCode -ne 0) {
    exit $rlmExitCode
}

$rlmCheckpoint = Join-Path $rlmRun "best.pt"
if (-not (Test-Path -LiteralPath $rlmCheckpoint)) {
    Write-SupervisorLog "ABORT: no RLM action checkpoint was produced."
    exit 5
}

Write-SupervisorLog "Running the held-out RLM evaluation."
Push-Location $root
try {
    $evalExitCode = Invoke-PythonLogged @(
        "scripts\run_rlm_eval.py",
        "--current-base-checkpoint", "runs\fdt_chat_language_ready\best.pt",
        "--current-rlm-checkpoint", "runs\fdt_rlm_language_ready_action\best.pt",
        "--tasks", "artifacts\rlm_tasks_heldout_1k.jsonl",
        "--per-family", "20",
        "--direct-tokens", "24",
        "--action-tokens", "48",
        "--device", "cuda",
        "--output-stem", "fdt_language_ready_rlm_heldout"
    )
} finally {
    Pop-Location
}
Write-SupervisorLog "Held-out RLM evaluation exited with code $evalExitCode."
exit $evalExitCode
