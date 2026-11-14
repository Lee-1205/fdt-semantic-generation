$Run = "<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM\runs\fdt_dual_memory_v1_1b_20260821_r3_continuous"
$Host.UI.RawUI.WindowTitle = "FDT Dual Memory V1 - 1B Training"

while ($true) {
    Clear-Host
    Write-Host "FDT Dual Memory V1 - continuous 1B-token training" -ForegroundColor Cyan
    Write-Host "Run: $Run"
    $PidPath = Join-Path $Run "train.pid"
    $LogPath = Join-Path $Run "training_log.jsonl"
    $ErrorPath = Join-Path $Run "stderr.log"
    if (Test-Path -LiteralPath $PidPath) {
        $TrainerPid = [int](Get-Content -LiteralPath $PidPath)
        $Process = Get-Process -Id $TrainerPid -ErrorAction SilentlyContinue
        Write-Host ("Status: " + $(if ($Process) { "RUNNING" } else { "STOPPED" })) -ForegroundColor $(if ($Process) { "Green" } else { "Yellow" })
        Write-Host "PID: $TrainerPid"
    }
    Write-Host ("C: free: {0:N2} GiB" -f ((Get-PSDrive C).Free / 1GB))
    Write-Host ""
    if (Test-Path -LiteralPath $LogPath) {
        Get-Content -LiteralPath $LogPath -Tail 8
    } else {
        Write-Host "Initializing model and fixed validation baseline..."
    }
    if ((Test-Path -LiteralPath $ErrorPath) -and (Get-Item -LiteralPath $ErrorPath).Length -gt 0) {
        Write-Host "`nRecent stderr:" -ForegroundColor Yellow
        Get-Content -LiteralPath $ErrorPath -Tail 4
    }
    Write-Host "`nTo request a safe stop, create STOP_REQUESTED in the run folder." -ForegroundColor DarkGray
    Start-Sleep -Seconds 5
}
