#Requires -Version 5.1
<#
.SYNOPSIS
    Build the LLM-Xray frontend and report packaging options.

.DESCRIPTION
    Runs the version consistency check, then builds apps/desktop with Vite into
    apps/desktop/dist. PyInstaller packaging is deliberately not automated here:
    it is slow and needs manual review of bundled assets and paths. The script
    prints the suggested manual packaging workflow at the end.

    This script does not start a server and does not call LM Studio.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/build.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$Root = (Resolve-Path -LiteralPath $Root).Path
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

# npm writes progress and warnings to stderr; Windows PowerShell 5.1 would treat
# that as a terminating error under EAP=Stop. Check the exit code explicitly.
function Invoke-Native {
    param(
        [scriptblock]$Command,
        [string]$Label
    )
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Command
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) { throw "$Label failed with exit code $code." }
}

Write-Output "[build] repository: $Root"

# esbuild writes temporary files under TEMP; some Windows security setups deny
# cleanup there. Keep the compiler's scratch space inside the repository.
$TempDir = Join-Path $Root 'data\tmp'
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
$env:TEMP = $TempDir
$env:TMP = $TempDir

# ------------------------------------------------- 1. version consistency ----
Write-Step 'Checking version consistency'
$hostExe = (Get-Process -Id $PID).Path
$checkScript = Join-Path $PSScriptRoot 'check-version.ps1'
$checkProc = Start-Process -FilePath $hostExe `
    -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$checkScript`"") `
    -Wait -PassThru -NoNewWindow
if ($checkProc.ExitCode -ne 0) {
    throw "Version consistency check failed (exit $($checkProc.ExitCode))."
}

# ------------------------------------------------------------ 2. frontend ----
Write-Step 'Building frontend (npm run build)'
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw 'Node.js 22 was not found on PATH.'
}
$npm = Get-NpmCommand
Push-Location $DesktopDir
try {
    if (-not (Test-Path -LiteralPath (Join-Path $DesktopDir 'node_modules') -PathType Container)) {
        Write-Output '[build] node_modules missing, running npm ci'
        Invoke-Native -Command { & $npm ci } -Label 'npm ci'
    }
    Invoke-Native -Command { & $npm run build } -Label 'npm run build'
} finally {
    Pop-Location
}

$distDir = Join-Path $DesktopDir 'dist'
if (-not (Test-Path -LiteralPath $distDir -PathType Container)) {
    throw "Build did not produce $distDir"
}
$files = Get-ChildItem -LiteralPath $distDir -Recurse -File
$totalBytes = ($files | Measure-Object -Property Length -Sum).Sum
Write-Output ("[build] dist: {0} file(s), {1:N1} MB" -f $files.Count, ($totalBytes / 1MB))

# --------------------------------------------------------- 3. packaging ----
Write-Step 'Packaging (not automated)'
Write-Output @'
PyInstaller packaging is intentionally NOT run by this script: it takes a long
time, needs a clean environment, and bundled asset paths must be reviewed by a
human before shipping.

Suggested manual workflow (review before use):

  1. Activate the project virtualenv and install the packager:
         pip install pyinstaller

  2. Build the frontend first:
         powershell -ExecutionPolicy Bypass -File scripts\build.ps1

  3. Run PyInstaller with an explicit spec file that includes:
         - desktop/launcher.py as the entry point
         - apps/desktop/dist as bundled static assets
         - docs/ and memory/ if they should ship with the app

  4. Smoke test the produced executable on a clean machine:
         - backend starts on 127.0.0.1:8765
         - the window opens and can reach LM Studio
         - docs/index.html opens from the bundled path
         - no write attempts into the installation directory (use data/)
'@

Write-Output ''
Write-Output "[build] frontend build complete: $distDir"
Write-Output '[build] run the desktop app with: python desktop\launcher.py'
