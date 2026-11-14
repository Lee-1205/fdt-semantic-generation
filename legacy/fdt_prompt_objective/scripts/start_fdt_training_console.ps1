param(
    [string]$Python = "python",
    [int]$Port = 7860,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$App = Join-Path $Root "apps\fdt_training_console\server.py"
$StateDir = Join-Path $Root "runs\fdt_training_console"
$PidFile = Join-Path $StateDir "app.pid"
$ServerFile = Join-Path $StateDir "app_server.json"

foreach ($Path in @($Root, $App, $StateDir, $PidFile, $ServerFile)) {
    $FullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $FullPath.StartsWith("C:\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "FDT Training Console permits only C: paths: $FullPath"
    }
}

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
if (Test-Path -LiteralPath $PidFile) {
    $ExistingPid = [int](Get-Content -LiteralPath $PidFile -Raw)
    $Existing = Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue
    if ($Existing -and (Test-Path -LiteralPath $ServerFile)) {
        $Server = Get-Content -LiteralPath $ServerFile -Raw | ConvertFrom-Json
        $Url = [string]$Server.url
        if (-not $NoBrowser) { Start-Process $Url | Out-Null }
        Write-Output "FDT Training Console is already running at $Url"
        exit 0
    }
    Remove-Item -LiteralPath $PidFile -Force
}

while (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    $Port += 1
}
$Stdout = Join-Path $StateDir "app.stdout.log"
$Stderr = Join-Path $StateDir "app.stderr.log"
$Process = Start-Process -FilePath $Python -ArgumentList @(
    $App,
    "--host", "127.0.0.1",
    "--port", [string]$Port
) -WorkingDirectory $Root -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
$Process.Id | Set-Content -LiteralPath $PidFile -Encoding ascii
$Url = "http://127.0.0.1:$Port"
[ordered]@{
    pid = $Process.Id
    url = $Url
    started_at = (Get-Date).ToString("o")
    app = $App
    app_sha256 = (Get-FileHash -LiteralPath $App -Algorithm SHA256).Hash
    stdout = $Stdout
    stderr = $Stderr
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $ServerFile -Encoding utf8

$Ready = $false
for ($Attempt = 0; $Attempt -lt 40; $Attempt++) {
    Start-Sleep -Milliseconds 250
    if (-not (Get-Process -Id $Process.Id -ErrorAction SilentlyContinue)) { break }
    try {
        $Response = Invoke-WebRequest -Uri "$Url/api/status" -UseBasicParsing -TimeoutSec 2
        if ($Response.StatusCode -eq 200) { $Ready = $true; break }
    } catch {
    }
}
if (-not $Ready) {
    throw "FDT Training Console did not become ready. See $Stderr"
}
if (-not $NoBrowser) { Start-Process $Url | Out-Null }
Write-Output "FDT Training Console started at $Url (PID $($Process.Id))"
