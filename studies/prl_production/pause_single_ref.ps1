param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$campaignRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $campaignRoot "..\..")).Path
$runtimeRoot = Join-Path $campaignRoot "manifests\runtime"
$recordPath = Join-Path $runtimeRoot "active_processes.json"
if (-not (Test-Path -LiteralPath $recordPath)) {
    throw "No active-process record exists at $recordPath"
}
$record = Get-Content -Raw -Encoding UTF8 -LiteralPath $recordPath | ConvertFrom-Json
$runId = [string]$record.run_id
if ($runId -notmatch "^[0-9a-f]{16}$") {
    throw "The active-process record has no valid run ID."
}
$statePath = Join-Path $runtimeRoot "single_ref_$($runId)_state.json"
$runner = Get-Process -Id ([int]$record.runner_pid) -ErrorAction SilentlyContinue
if ($null -eq $runner) {
    Write-Output "The recorded runner is no longer active."
    exit 0
}
if (-not $Force) {
    throw "Re-run with -Force to terminate the recorded production process tree after its latest atomic checkpoint."
}

# This script targets only the PID recorded by run_single_ref.ps1. /T includes
# its worker children; point files written before termination remain valid and
# the next launch resumes incomplete graph indices.
& taskkill.exe /PID $runner.Id /T /F

if (Test-Path -LiteralPath $statePath) {
    $state = Get-Content -Raw -Encoding UTF8 -LiteralPath $statePath | ConvertFrom-Json
    $state.state = "interrupted"
    $state.updated_unix = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
    $tempState = "$statePath.$([guid]::NewGuid().ToString('N')).tmp"
    $state | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $tempState
    Move-Item -LiteralPath $tempState -Destination $statePath -Force
}

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
Push-Location $repoRoot
try {
    & $python -m studies.prl_production.single_ref.monitor --run-id $runId
}
finally {
    Pop-Location
}
$record.state = "interrupted"
$record.stopped = (Get-Date).ToString("o")
$tempRecord = "$recordPath.$([guid]::NewGuid().ToString('N')).tmp"
$record | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $tempRecord
Move-Item -LiteralPath $tempRecord -Destination $recordPath -Force
Write-Output "Paused runner PID $($runner.Id). Data through the last atomic checkpoint are preserved."
