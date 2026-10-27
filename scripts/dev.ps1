#Requires -Version 5.1
<#
.SYNOPSIS
    Start the LLM-Xray development environment: FastAPI backend + Vite dev server.

.DESCRIPTION
    Starts uvicorn on 127.0.0.1:8765 (backend/run.py when available, otherwise the
    FastAPI module is auto-detected) and Vite on 127.0.0.1:5173. Waits until the
    backend health endpoint answers, prints both URLs, then keeps the Vite server
    in the foreground. Press Ctrl+C to stop Vite; the backend is stopped with it.

    For the packaged/desktop workflow use scripts/build.ps1 and then
    "python desktop/launcher.py".

.PARAMETER BackendPort
    Backend port. Default 8765.

.PARAMETER FrontendPort
    Vite port. Default 5173.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/dev.ps1
#>
[CmdletBinding()]
param(
    [int]$BackendPort = 8765,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$Root = (Resolve-Path -LiteralPath $Root).Path
$BackendDir = Join-Path $Root 'backend'
$DesktopDir = Join-Path $Root 'apps\desktop'

function Write-Step {
    param([string]$Message)
    Write-Output ''
    Write-Output "==> $Message"
}

function Get-NpmCommand {
    $cmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cmd = Get-Command npm -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw 'npm was not found on PATH. Install Node.js 22 and retry.'
}

function Find-FastApiTarget {
    param([string]$BackendDirectory)

    $candidates = @()
    if (Test-Path -LiteralPath (Join-Path $BackendDirectory 'run.py') -PathType Leaf) {
        return @{ Kind = 'script'; Value = 'run.py' }
    }

    $pyFiles = Get-ChildItem -LiteralPath (Join-Path $BackendDirectory 'xray') -Filter '*.py' -File -ErrorAction SilentlyContinue
    foreach ($file in $pyFiles) {
        $text = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
        $match = [regex]::Match($text, '(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*FastAPI\(')
        if ($match.Success) {
            $appName = $match.Groups[1].Value
            $module = 'xray.' + [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
            $candidates += "$module`:$appName"
        }
    }

    if ($candidates.Count -eq 0) {
        # The desktop launcher starts ``uvicorn xray.app:app``; use it as the last resort.
        if (Test-Path -LiteralPath (Join-Path $BackendDirectory 'xray\app.py') -PathType Leaf) {
            return @{ Kind = 'uvicorn'; Value = 'xray.app:app' }
        }
        return $null
    }
    return @{ Kind = 'uvicorn'; Value = $candidates[0] }
}

Write-Output "[dev] repository: $Root"

# ------------------------------------------------------------- tool checks ----
$python = Join-Path $Root '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $python -PathType Leaf) {
    Write-Output "[dev] python: $python"
} else {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw 'Python 3.11 was not found. Create the virtualenv described in README.md.'
    }
    $python = 'python'
    Write-Output '[dev] python: system python (create .venv to pin the environment)'
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw 'Node.js 22 was not found on PATH.'
}
$npm = Get-NpmCommand

if (-not (Test-Path -LiteralPath (Join-Path $DesktopDir 'node_modules') -PathType Container)) {
    Write-Output '[dev] node_modules missing, running npm install (one time)'
    Push-Location $DesktopDir
    try {
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try { & $npm install } finally { $ErrorActionPreference = $previous }
        if ($LASTEXITCODE -ne 0) { throw "npm install failed with exit code $LASTEXITCODE." }
    } finally {
        Pop-Location
    }
}

# ------------------------------------------------------------------ backend ----
$target = Find-FastApiTarget -BackendDirectory $BackendDir
if ($null -eq $target) {
    throw 'Could not find a FastAPI entry point (backend/run.py or an "app = FastAPI(...)" module).'
}

if (-not (Test-Path -LiteralPath (Join-Path $BackendDir 'xray') -PathType Container)) {
    throw "Backend package not found: $BackendDir\xray"
}

Write-Step 'Starting backend'
$backendProc = $null
try {
    if ($target.Kind -eq 'script') {
        Write-Output "[dev] backend: python $($target.Value) (127.0.0.1:$BackendPort)"
        $backendProc = Start-Process -FilePath $python -WorkingDirectory $BackendDir `
            -ArgumentList @($target.Value) -PassThru
    } else {
        Write-Output "[dev] backend: uvicorn $($target.Value) --host 127.0.0.1 --port $BackendPort --reload"
        $backendProc = Start-Process -FilePath $python -WorkingDirectory $BackendDir `
            -ArgumentList @('-m', 'uvicorn', $target.Value, '--host', '127.0.0.1', '--port', "$BackendPort", '--reload') -PassThru
    }

    $healthUrl = "http://127.0.0.1:$BackendPort/api/health"
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        if ($backendProc.HasExited) {
            throw "Backend exited during startup (exit code $($backendProc.ExitCode)). Check the backend log above."
        }
        try {
            $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($response.StatusCode -eq 200) { $ready = $true; break }
        } catch {
            # not ready yet
        }
    }
    if (-not $ready) {
        throw "Backend did not become healthy at $healthUrl within 30 seconds."
    }
    Write-Output "[dev] backend ready: http://127.0.0.1:$BackendPort/api/health"

    # ------------------------------------------------------------ frontend ----
    Write-Step 'Starting Vite'
    $viteProc = Start-Process -FilePath $npm -WorkingDirectory $DesktopDir `
        -ArgumentList @('run', 'dev', '--', '--host', '127.0.0.1', '--port', "$FrontendPort") -PassThru

    Write-Output ''
    Write-Output ("  Dev server : http://127.0.0.1:{0}" -f $FrontendPort)
    Write-Output ("  API        : http://127.0.0.1:{0}" -f $BackendPort)
    Write-Output ("  Health     : http://127.0.0.1:{0}/api/health" -f $BackendPort)
    Write-Output '  Docs page  : docs/index.html (open directly in a browser)'
    Write-Output ''
    Write-Output '  Press Ctrl+C to stop Vite. The backend process is stopped with it.'
    Write-Output '  Production: run scripts\build.ps1, then python desktop\launcher.py'
    Write-Output ''

    Wait-Process -Id $viteProc.Id
} finally {
    if ($null -ne $backendProc -and -not $backendProc.HasExited) {
        Write-Output ''
        Write-Output '[dev] stopping backend'
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'SilentlyContinue'
        try { & taskkill /PID $backendProc.Id /T /F 2>$null | Out-Null } finally { $ErrorActionPreference = $previous }
    }
}
