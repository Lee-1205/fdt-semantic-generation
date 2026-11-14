$ErrorActionPreference = "Stop"

$Root = "<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM"
$Server = Join-Path $Root "apps\fdt_anchor_exact_console\server.py"
$StateDir = Join-Path $Root "runs\fdt_anchor_exact_console"
$ServerPidPath = Join-Path $StateDir "server.pid"
$Running = $false

if (Test-Path -LiteralPath $ServerPidPath) {
    $SavedPid = [int](Get-Content -LiteralPath $ServerPidPath)
    $SavedProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$SavedPid" -ErrorAction SilentlyContinue
    $Running = $null -ne $SavedProcess -and $SavedProcess.CommandLine -like "*$Server*"
}
if (-not $Running) {
    New-Item -ItemType Directory -Path $StateDir -Force | Out-Null
    $Python = (Get-Command python -ErrorAction Stop).Source
    $Console = Start-Process -FilePath $Python -ArgumentList @("-u", $Server, "--port", "7861") -WorkingDirectory $Root -RedirectStandardOutput (Join-Path $StateDir "stdout.log") -RedirectStandardError (Join-Path $StateDir "stderr.log") -WindowStyle Hidden -PassThru
    $Console.Id | Set-Content -LiteralPath $ServerPidPath -Encoding ascii
    $Deadline = (Get-Date).AddSeconds(15)
    do {
        Start-Sleep -Milliseconds 250
        $Listening = Get-NetTCPConnection -LocalPort 7861 -State Listen -ErrorAction SilentlyContinue
    } while (-not $Listening -and (Get-Date) -lt $Deadline)
    if (-not $Listening) {
        throw "FDT console did not start"
    }
}
Start-Process "http://127.0.0.1:7861/"
