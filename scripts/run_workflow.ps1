#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet("setup", "discovery", "ingest", "features", "targets", "baselines", "oil-stress", "all")]
    [string[]] $Stage = @("setup"),

    [string] $StartMonth = "2000-01",
    [string] $EndMonth = "",
    [string] $StartQuarter = "2000-Q1",
    [string] $EndQuarter = "",

    [ValidateSet("jsonstat", "sdmx-csv")]
    [string] $Format = "sdmx-csv",

    [string] $PythonExe = "",
    [switch] $ForceRefresh,
    [switch] $WithPolars,
    [switch] $SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$SupportedPythonVersions = @("3.10", "3.11")

function Get-PythonInfo {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Executable,

        [string[]] $PrefixArgs = @()
    )

    try {
        $probe = @($(& $Executable @PrefixArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}'); print(sys.executable)" 2>$null))
        if ($LASTEXITCODE -ne 0 -or $probe.Count -lt 2) {
            return $null
        }

        return [pscustomobject]@{
            Command = $Executable
            Args = $PrefixArgs
            Executable = $probe[1]
            Version = $probe[0]
        }
    }
    catch {
        return $null
    }

    return $null
}

function Find-SupportedPython {
    $localAppData = [Environment]::GetFolderPath("LocalApplicationData")
    $programFiles = [Environment]::GetFolderPath("ProgramFiles")
    $userProfile = [Environment]::GetFolderPath("UserProfile")
    $repoRootPath = [string] $RepoRoot
    $repoUserRoot = $null
    if ($repoRootPath -match "^[A-Za-z]:\\Users\\[^\\]+") {
        $repoUserRoot = $Matches[0]
    }

    $codexRuntimeCandidates = @(
        (Join-Path $userProfile ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe")
    )

    if ($repoUserRoot) {
        $codexRuntimeCandidates += (Join-Path $repoUserRoot ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe")
    }

    $candidates = @()

    if ($PythonExe) {
        $candidates += @{ Executable = $PythonExe; Args = @() }
    }

    $candidates += @(
        @{ Executable = (Join-Path $RepoRoot ".venv\Scripts\python.exe"); Args = @() },
        @{ Executable = (Join-Path $RepoRoot "venv\Scripts\python.exe"); Args = @() },
        @{ Executable = "py"; Args = @("-3.11") },
        @{ Executable = "py"; Args = @("-3.10") },
        @{ Executable = "python3.11"; Args = @() },
        @{ Executable = "python3.10"; Args = @() },
        @{ Executable = "python"; Args = @() },
        @{ Executable = (Join-Path $localAppData "Programs\Python\Python311\python.exe"); Args = @() },
        @{ Executable = (Join-Path $localAppData "Programs\Python\Python310\python.exe"); Args = @() },
        @{ Executable = (Join-Path $programFiles "Python311\python.exe"); Args = @() }
    )

    foreach ($runtimePython in $codexRuntimeCandidates) {
        $candidates += @{ Executable = $runtimePython; Args = @() }
    }

    $unsupportedCandidates = @()
    foreach ($candidate in $candidates) {
        $result = Get-PythonInfo -Executable $candidate.Executable -PrefixArgs $candidate.Args
        if ($null -eq $result) {
            continue
        }

        if ($SupportedPythonVersions -contains $result.Version) {
            return $result
        }

        $unsupportedCandidates += $result
    }

    if ($unsupportedCandidates.Count -gt 0) {
        $found = ($unsupportedCandidates | ForEach-Object { "$($_.Version) at $($_.Executable)" }) -join "; "
        throw "Could not find a supported Python version. This project requires Python 3.10 or 3.11. Found other Python versions instead: $found. Install Python 3.10 or 3.11, or rerun with -PythonExe <full-path-to-python.exe>."
    }

    throw "Could not find a supported Python version. This project requires Python 3.10 or 3.11. Install Python 3.10 or 3.11, add it to PATH, or rerun with -PythonExe <full-path-to-python.exe>."
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string] $FilePath,

        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    Write-Host "> $FilePath $($Arguments -join ' ')"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function Ensure-Venv {
    if (-not (Test-Path $VenvPython)) {
        $basePython = Find-SupportedPython
        Write-Host "Creating .venv with $($basePython.Executable)"
        Invoke-Checked -FilePath $basePython.Executable -Arguments @("-m", "venv", $VenvDir)
    }

    $venvProbe = Get-PythonInfo -Executable $VenvPython
    if ($null -eq $venvProbe -or -not ($SupportedPythonVersions -contains $venvProbe.Version)) {
        throw ".venv exists, but $VenvPython is not Python 3.10 or 3.11. Remove .venv and rerun this script with a supported Python version available."
    }
}

function Install-Project {
    if ($SkipInstall) {
        return
    }

    Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "-e", ".[dev]")
    if ($WithPolars) {
        Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "-e", ".[polars]")
    }
}

function Invoke-Module {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Module,

        [string[]] $ModuleArgs = @()
    )

    Invoke-Checked -FilePath $VenvPython -Arguments (@("-m", $Module) + $ModuleArgs)
}

Set-Location $RepoRoot
Ensure-Venv
Install-Project

$expandedStages = if ($Stage -contains "all") {
    @("ingest", "features", "targets", "baselines", "oil-stress")
}
else {
    $Stage | Where-Object { $_ -ne "setup" }
}

foreach ($item in $expandedStages) {
    switch ($item) {
        "discovery" {
            Invoke-Module -Module "data_access.discovery_runner"
        }
        "ingest" {
            $args = @("--start", $StartMonth, "--format", $Format)
            if ($EndMonth) { $args += @("--end", $EndMonth) }
            if ($ForceRefresh) { $args += "--force-refresh" }
            Invoke-Module -Module "data_access.pull_eurostat" -ModuleArgs $args
        }
        "features" {
            $args = @("--input", "data_processed/eurostat/selected_series_monthly.parquet")
            if ($StartMonth) { $args += @("--start", $StartMonth) }
            if ($EndMonth) { $args += @("--end", $EndMonth) }
            Invoke-Module -Module "features.monthly_features" -ModuleArgs $args
        }
        "targets" {
            $args = @("--start", $StartQuarter)
            if ($EndQuarter) { $args += @("--end", $EndQuarter) }
            if ($ForceRefresh) { $args += "--force-refresh" }
            Invoke-Module -Module "features.targets" -ModuleArgs $args
        }
        "baselines" {
            Invoke-Module -Module "models.baselines" -ModuleArgs @(
                "--features", "data_processed/features/monthly_features_long.csv",
                "--targets", "data_processed/targets/monthly_bridge_targets.csv"
            )
        }
        "oil-stress" {
            $args = @("--start", $StartMonth, "--format", $Format)
            if ($EndMonth) { $args += @("--end", $EndMonth) }
            if ($ForceRefresh) { $args += "--force-refresh" }
            Invoke-Module -Module "features.oil_stress" -ModuleArgs $args
        }
    }
}
