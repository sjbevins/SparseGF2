param(
    [ValidateSet("smoke", "pilot", "production")]
    [string]$Profile = "smoke",
    [ValidateRange(1, 32)]
    [int]$Workers = 8,
    [ValidateSet("Idle", "BelowNormal", "Normal")]
    [string]$Priority = "BelowNormal",
    [switch]$ConfirmProduction,
    [switch]$Foreground,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$campaignRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $campaignRoot "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Repository interpreter not found: $python"
}
if ($Profile -eq "production" -and -not $ConfirmProduction) {
    throw "Production requires -ConfirmProduction after protocol and Git review."
}

$runnerArgs = @(
    "-u",
    "-m",
    "studies.prl_production.single_ref.run",
    "--profile", $Profile,
    "--workers", "$Workers"
)
if ($Profile -eq "production") {
    $runnerArgs += "--confirm-production"
}
$runnerArgs += $ExtraArgs

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logRoot = Join-Path $campaignRoot "logs"
$runtimeRoot = Join-Path $campaignRoot "manifests\runtime"
New-Item -ItemType Directory -Force -Path $logRoot, $runtimeRoot | Out-Null
$stdout = Join-Path $logRoot "single_ref_$($Profile)_$($stamp).out.log"
$stderr = Join-Path $logRoot "single_ref_$($Profile)_$($stamp).err.log"
$monitorStdout = Join-Path $logRoot "single_ref_$($Profile)_$($stamp).monitor.out.log"
$monitorStderr = Join-Path $logRoot "single_ref_$($Profile)_$($stamp).monitor.err.log"

if ($Foreground) {
    Push-Location $repoRoot
    try {
        & $python @runnerArgs
        exit $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

# Resolve the content-addressed run before detaching so the monitor and pause
# command can target this run even when a newer smoke manifest exists.
$resolveArgs = @($runnerArgs) + "--print-run-id"
Push-Location $repoRoot
try {
    $resolvedOutput = & $python @resolveArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve the single-reference run ID."
    }
}
finally {
    Pop-Location
}
$runId = ([string]($resolvedOutput | Select-Object -Last 1)).Trim()
if ($runId -notmatch "^[0-9a-f]{16}$") {
    throw "Unexpected single-reference run ID: $runId"
}
$manifestPath = Join-Path $runtimeRoot "single_ref_$($runId)_manifest.json"
$statePath = Join-Path $runtimeRoot "single_ref_$($runId)_state.json"

$runnerStart = @{
    FilePath = $python
    ArgumentList = $runnerArgs
    WorkingDirectory = $repoRoot
    RedirectStandardOutput = $stdout
    RedirectStandardError = $stderr
    WindowStyle = "Hidden"
    PassThru = $true
}
$runner = Start-Process @runnerStart
$runner.PriorityClass = $Priority

$monitorArgs = @(
    "-u",
    "-m",
    "studies.prl_production.single_ref.monitor",
    "--run-id", $runId,
    "--watch",
    "--interval", "60"
)
Start-Sleep -Seconds 2
$monitorStart = @{
    FilePath = $python
    ArgumentList = $monitorArgs
    WorkingDirectory = $repoRoot
    RedirectStandardOutput = $monitorStdout
    RedirectStandardError = $monitorStderr
    WindowStyle = "Hidden"
    PassThru = $true
}
$monitor = Start-Process @monitorStart
$monitor.PriorityClass = "BelowNormal"

$processRecord = @{
    profile = $Profile
    run_id = $runId
    workers = $Workers
    priority = $Priority
    runner_pid = $runner.Id
    monitor_pid = $monitor.Id
    manifest_path = $manifestPath
    state_path = $statePath
    stdout = $stdout
    stderr = $stderr
    monitor_stdout = $monitorStdout
    monitor_stderr = $monitorStderr
    started = (Get-Date).ToString("o")
}
$recordPath = Join-Path $runtimeRoot "active_processes.json"
$tempRecordPath = "$recordPath.$([guid]::NewGuid().ToString('N')).tmp"
$processRecord |
    ConvertTo-Json |
    Set-Content -Encoding UTF8 -LiteralPath $tempRecordPath
Move-Item -LiteralPath $tempRecordPath -Destination $recordPath -Force

Write-Output "run_id=$runId"
Write-Output "runner_pid=$($runner.Id)"
Write-Output "monitor_pid=$($monitor.Id)"
Write-Output "stdout=$stdout"
Write-Output "stderr=$stderr"
Write-Output "monitor_stdout=$monitorStdout"
Write-Output "monitor_stderr=$monitorStderr"
Write-Output "status=$(Join-Path $campaignRoot 'STATUS.md')"
