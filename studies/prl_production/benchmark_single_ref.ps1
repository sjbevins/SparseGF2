param(
    [ValidateRange(5, 1000000)]
    [int]$N = 64,
    [string]$MeasurementRate = "0",
    [ValidateRange(1, 1000000)]
    [int]$QMax = 1,
    [ValidateRange(0, 1000000)]
    [int]$QScramble = 1,
    [ValidateRange(1, 1000000000)]
    [int]$Circuits = 16,
    [string]$Workers = "1,2",
    [ValidateRange(1, 20)]
    [int]$Repetitions = 3,
    [ValidateRange(0.001, 3600.0)]
    [double]$TargetTaskSeconds = 0.5,
    [ValidateRange(0.001, 3600.0)]
    [double]$MaxEstimatedSecondsPerRepeat = 20.0,
    [ValidateSet("batch", "scalar")]
    [string]$Execution = "batch",
    [switch]$NoHybrid,
    [string]$EdgesNpy = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$campaignRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $campaignRoot "..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Repository interpreter not found: $python"
}

# These values must exist before Python imports NumPy or Numba.  The trajectory
# kernel itself is single-threaded; production parallelism is across processes.
$env:NUMBA_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$workerValues = @()
foreach ($value in ($Workers -split ",")) {
    $parsed = 0
    if (-not [int]::TryParse($value.Trim(), [ref]$parsed) -or $parsed -lt 1 -or $parsed -gt 32) {
        throw "Workers must be a comma-separated list of integers in [1, 32]."
    }
    $workerValues += "$parsed"
}
if ($workerValues.Count -eq 0) {
    throw "Workers must not be empty."
}

$invariant = [Globalization.CultureInfo]::InvariantCulture
$BenchmarkArgs = @(
    "--n", "$N",
    "--p", $MeasurementRate,
    "--q-max", "$QMax",
    "--q-scramble", "$QScramble",
    "--circuits", "$Circuits",
    "--workers"
) + $workerValues + @(
    "--repetitions", "$Repetitions",
    "--target-task-seconds", $TargetTaskSeconds.ToString($invariant),
    "--max-estimated-seconds-per-repeat", $MaxEstimatedSecondsPerRepeat.ToString($invariant),
    "--execution", $Execution
)
if ($NoHybrid) {
    $BenchmarkArgs += "--no-hybrid"
}
if ($EdgesNpy) {
    $BenchmarkArgs += @("--edges-npy", $EdgesNpy)
}
if ($Output) {
    $BenchmarkArgs += @("--output", $Output)
}

Push-Location $repoRoot
try {
    & $python -u -m studies.prl_production.single_ref.raw_tau.benchmark @BenchmarkArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
