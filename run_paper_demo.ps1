param(
    [string]$Profile = "us_extra_trees_daily",
    [string]$SessionDate = (Get-Date -Format "yyyy-MM-dd"),
    [string]$LedgerPath = "artifacts/paper_demo/paper_ledger.sqlite3",
    [string]$ArtifactRoot = "artifacts",
    [Nullable[double]]$ExecutionEquityCap = $null,
    [Nullable[double]]$MaxOrderNotional = $null,
    [Nullable[double]]$MaxTotalNotional = $null,
    [Nullable[int]]$MaxTotalOrders = $null,
    [switch]$Execute,
    [switch]$AllowUnhealthy,
    [switch]$SkipHealthcheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-JsonPythonModule {
    param(
        [string]$Module,
        [string[]]$Arguments
    )

    $command = @("-m", $Module) + $Arguments
    Write-Host ("python " + ($command -join " ")) -ForegroundColor DarkGray

    $output = & python @command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: python $($command -join ' ')"
    }

    $text = ($output -join "`n").Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        throw "Command returned empty output: python $($command -join ' ')"
    }

    $jsonStart = $text.IndexOf("{")
    $jsonEnd = $text.LastIndexOf("}")
    if ($jsonStart -lt 0 -or $jsonEnd -lt $jsonStart) {
        throw "Command did not emit a JSON payload: python $($command -join ' ')"
    }

    $jsonText = $text.Substring($jsonStart, $jsonEnd - $jsonStart + 1)
    try {
        $payload = $jsonText | ConvertFrom-Json
    }
    catch {
        throw "Command emitted invalid JSON payload: python $($command -join ' ')"
    }

    return [PSCustomObject]@{
        RawText = $text
        Ok = [bool]$payload.ok
        RunId = if ($payload.PSObject.Properties.Name -contains "run_id") { $payload.run_id } else { $null }
    }
}

Push-Location $RepoRoot
try {
    $env:PYTHONPATH = "src"
    $env:PYTHONWARNINGS = "ignore::FutureWarning"

    if (-not $SkipHealthcheck) {
        Write-Step "Healthcheck"
        $health = Invoke-JsonPythonModule -Module "stockmachine.apps.paper_daily" -Arguments @(
            "healthcheck",
            "--ledger-path", $LedgerPath
        )
        Write-Host $health.RawText
        if (-not $health.ok) {
            throw "Healthcheck failed."
        }
        Write-Host "Healthcheck OK" -ForegroundColor Green
    }

    $smokeRunName = "{0}-smoke-{1}" -f $Profile, (Get-Date -Format "yyyyMMdd-HHmmss")
    $smokeArgs = @(
        "--strategy-profile", $Profile,
        "--session-date", $SessionDate,
        "--run-name", $smokeRunName,
        "--artifact-root", $ArtifactRoot,
        "--ledger-path", $LedgerPath
    )
    if ($AllowUnhealthy) {
        $smokeArgs += "--allow-unhealthy"
    }

    Write-Step "Smoke Dry-Run"
    $smoke = Invoke-JsonPythonModule -Module "stockmachine.apps.paper_smoke" -Arguments $smokeArgs
    Write-Host $smoke.RawText
    if (-not $smoke.ok) {
        throw "Smoke failed."
    }

    Write-Host ("Smoke OK | run_id={0}" -f $smoke.RunId) -ForegroundColor Green

    if (-not $Execute) {
        Write-Step "Done"
        Write-Host "Dry-run complete. To place paper orders, rerun with -Execute." -ForegroundColor Yellow
        return
    }

    $runArgs = @(
        "run",
        "--strategy-profile", $Profile,
        "--session-date", $SessionDate,
        "--artifact-root", $ArtifactRoot,
        "--ledger-path", $LedgerPath,
        "--execute"
    )
    if ($null -ne $ExecutionEquityCap) {
        $runArgs += @(
            "--execution-equity-cap",
            $ExecutionEquityCap.Value.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        )
    }
    if ($null -ne $MaxOrderNotional) {
        $runArgs += @(
            "--max-order-notional",
            $MaxOrderNotional.Value.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        )
    }
    if ($null -ne $MaxTotalNotional) {
        $runArgs += @(
            "--max-total-notional",
            $MaxTotalNotional.Value.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        )
    }
    if ($null -ne $MaxTotalOrders) {
        $runArgs += @(
            "--max-total-orders",
            $MaxTotalOrders.Value.ToString()
        )
    }
    if ($AllowUnhealthy) {
        $runArgs += "--allow-unhealthy"
    }

    Write-Step "Paper Execute"
    $run = Invoke-JsonPythonModule -Module "stockmachine.apps.paper_daily" -Arguments $runArgs
    Write-Host $run.RawText
    if (-not $run.ok) {
        throw "Paper execute failed."
    }

    Write-Host ("Paper execute OK | run_id={0}" -f $run.RunId) -ForegroundColor Green

    Write-Step "Next Checks"
    Write-Host "python -m stockmachine.apps.paper_ops latest-run"
    Write-Host "python -m stockmachine.apps.paper_reconcile latest-run --ledger $LedgerPath"
}
finally {
    Pop-Location
}
