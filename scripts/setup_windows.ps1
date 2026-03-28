$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$VenvPath = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    & $Command @Args
    if ($LASTEXITCODE -ne 0) {
        $joinedArgs = $Args -join " "
        throw "Command failed: $Command $joinedArgs"
    }
}

function Test-PythonCandidate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [Parameter(Mandatory = $true)]
        [string[]]$ProbeArgs
    )

    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        return $false
    }

    try {
        & $Command @ProbeArgs 1>$null 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Get-PythonInvocation {
    $candidates = @(
        @{ Command = "py"; Args = @("-3.11") },
        @{ Command = "py"; Args = @("-3") },
        @{ Command = "python" },
        @{ Command = "python3.13" },
        @{ Command = "python3.12" },
        @{ Command = "python3.11" }
    )

    foreach ($candidate in $candidates) {
        $probeArgs = @()
        if ($candidate.ContainsKey("Args")) {
            $probeArgs += $candidate.Args
        }
        $probeArgs += @(
            "-c",
            "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
        )

        if (Test-PythonCandidate -Command $candidate.Command -ProbeArgs $probeArgs) {
            return $candidate
        }
    }

    return $null
}

if (-not (Test-Path $VenvPython)) {
    $python = Get-PythonInvocation

    if ($null -ne $python) {
        $venvArgs = @()
        if ($python.ContainsKey("Args")) {
            $venvArgs += $python.Args
        }
        $venvArgs += @("-m", "venv", $VenvPath)
        Invoke-NativeCommand -Command $python.Command -Args $venvArgs
    }
    elseif (Get-Command uv -ErrorAction SilentlyContinue) {
        Invoke-NativeCommand -Command "uv" -Args @("venv", $VenvPath, "--python", "3.11", "--seed")
    }
    else {
        throw "No usable Python 3.11+ interpreter was found. Install Python 3.11+ or uv and retry."
    }
}

if (-not (Test-PythonCandidate -Command $VenvPython -ProbeArgs @("-m", "pip", "--version"))) {
    Invoke-NativeCommand -Command $VenvPython -Args @("-m", "ensurepip", "--upgrade")
}

Invoke-NativeCommand -Command $VenvPython -Args @("-m", "pip", "install", "--upgrade", "pip")
Invoke-NativeCommand -Command $VenvPython -Args @("-m", "pip", "install", "-e", ".[dev,audio]")

Write-Host "Setup complete."
Write-Host "Repo root: $RepoRoot"
Write-Host "Activate with: $ActivateScript"
Write-Host "Run with: $VenvPython -m cortana"
