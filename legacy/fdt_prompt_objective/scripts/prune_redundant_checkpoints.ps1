param(
    [string]$RunsRoot = "runs",
    [int]$KeepNumberedPerRun = 2,
    [switch]$Apply
)

$root = (Resolve-Path -LiteralPath $RunsRoot).Path
$candidates = @()

if ($KeepNumberedPerRun -lt 0) {
    throw "KeepNumberedPerRun must be non-negative."
}

$directories = @((Get-Item -LiteralPath $root)) + @(
    Get-ChildItem -LiteralPath $root -Directory -Recurse
)
foreach ($directory in $directories) {
    $hasRecoveryCheckpoint =
        (Test-Path -LiteralPath (Join-Path $directory.FullName "latest.pt")) -or
        (Test-Path -LiteralPath (Join-Path $directory.FullName "onk_ready.pt"))
    if (-not $hasRecoveryCheckpoint) {
        continue
    }

    $numbered = Get-ChildItem -LiteralPath $directory.FullName -File -Filter "step_*.pt" |
        Where-Object { $_.Name -match '^step_(\d+)\.pt$' } |
        Sort-Object { [int64]([regex]::Match($_.Name, '^step_(\d+)\.pt$').Groups[1].Value) }
    if ($numbered.Count -gt $KeepNumberedPerRun) {
        $candidates += $numbered | Select-Object -First ($numbered.Count - $KeepNumberedPerRun)
    }
}

$bytes = ($candidates | Measure-Object -Property Length -Sum).Sum
[pscustomobject]@{
    Mode = if ($Apply) { "APPLY" } else { "DRY_RUN" }
    RunsRoot = $root
    Files = $candidates.Count
    Gigabytes = [math]::Round($bytes / 1GB, 2)
}

if ($Apply) {
    foreach ($file in $candidates) {
        $resolved = $file.FullName
        if (-not $resolved.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to delete outside runs root: $resolved"
        }
        Remove-Item -LiteralPath $resolved -Force
    }
}
