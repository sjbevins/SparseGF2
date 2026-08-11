param(
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [switch]$Run,
    [string]$ConfirmExperimentId = "",
    [int]$Workers = 0
)

$ErrorActionPreference = "Stop"
$campaignRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $campaignRoot "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Repository interpreter not found: $python"
}
$configPath = (Resolve-Path -LiteralPath $Config).Path
if ($Workers -lt 0) {
    throw "Workers must be zero (use the config value) or a positive integer."
}
if ($Run -and -not $ConfirmExperimentId) {
    throw "-Run requires -ConfirmExperimentId from a reviewed plan."
}

# The trajectory kernel is single-threaded. Production parallelism is across
# worker processes, so native numerical libraries must not spawn nested pools.
$env:NUMBA_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$runnerArgs = @("--config", $configPath)
if ($Workers -gt 0) {
    $runnerArgs += @("--workers", "$Workers")
}
if ($Run) {
    $runnerArgs += @("--run", "--confirm-experiment-id", $ConfirmExperimentId)
}
else {
    $runnerArgs += "--plan"
}

$exitCode = 0
Push-Location $repoRoot
try {
    & $python -u -m studies.prl_production.single_ref.raw_tau.run @runnerArgs
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $exitCode
