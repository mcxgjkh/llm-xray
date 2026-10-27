#Requires -Version 5.1
<#
.SYNOPSIS
    Validate the release state and create an annotated git tag for the current version.

.DESCRIPTION
    Fails fast on the first problem. Steps:
      1. check version consistency (VERSION, package.json, CHANGELOG.md, docs/VERSIONS.md)
      2. require a clean git working tree and an existing git repository
      3. require that tag v<version> does not exist yet
      4. run the frontend production build (npm run build)
      5. run the backend test suite (python -m pytest -q) unless -SkipTests
      6. create an annotated tag v<version>; with -DryRun only print the plan

    Nothing is pushed unless -Push is given.

.PARAMETER DryRun
    Run every validation and print the tag command without creating the tag.

.PARAMETER SkipTests
    Skip the backend pytest step. Use only for documentation-only releases.

.PARAMETER Push
    Push the created tag to origin after a successful run.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/release.ps1 -DryRun

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/release.ps1
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$SkipTests,
    [switch]$Push
)

$ErrorActionPreference = 'Stop'

trap {
    Write-Output ''
    Write-Output "[release] ERROR: $($_.Exception.Message)"
    exit 1
}

$Root = Split-Path -Parent $PSScriptRoot
$Root = (Resolve-Path -LiteralPath $Root).Path
$DesktopDir = Join-Path $Root 'apps\desktop'
$BackendDir = Join-Path $Root 'backend'

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

function Assert-Tool {
    param([string]$Name, [string]$Hint)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found on PATH. $Hint"
    }
}

# Native tools (npm, pytest, git) write progress and warnings to stderr. Windows
# PowerShell 5.1 would turn that into a terminating error under EAP=Stop, so native
# calls run with a local Continue preference and the exit code is checked explicitly.
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

function Invoke-NativeCapture {
    param([scriptblock]$Command)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $Command 2>$null
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    return [pscustomobject]@{ Output = @($output); ExitCode = $code }
}

Write-Output "[release] repository: $Root"

# Keep esbuild scratch files inside the repository: some Windows security
# configurations block cleanup under the system TEMP directory.
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
    throw "Version consistency check failed (exit $($checkProc.ExitCode)). Fix versions before releasing."
}

$version = (Get-Content -LiteralPath (Join-Path $Root 'VERSION') -Raw -Encoding UTF8).Trim()
$tag = "v$version"
Write-Output "[release] version: $version -> tag: $tag"

# --------------------------------------------------------- 2. git checks ----
Write-Step 'Checking git working tree'

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'git was not found on PATH.'
}
$inside = Invoke-NativeCapture { & git -C $Root rev-parse --is-inside-work-tree }
if ($inside.ExitCode -ne 0 -or ("$($inside.Output)".Trim() -ne 'true')) {
    throw "Not a git repository: $Root"
}

$statusResult = Invoke-NativeCapture { & git -C $Root status --porcelain }
if ($statusResult.ExitCode -ne 0) {
    throw 'git status failed.'
}
$status = @($statusResult.Output)
if ($status.Count -gt 0) {
    Write-Output 'Uncommitted changes:'
    $status | ForEach-Object { Write-Output "  $_" }
    throw 'Working tree is not clean. Commit or stash changes before releasing.'
}
Write-Output '[release] working tree is clean.'

# ------------------------------------------------------ 3. tag availability ----
Write-Step "Checking tag $tag"

$localTag = Invoke-NativeCapture { & git -C $Root rev-parse -q --verify "refs/tags/$tag" }
$tagExists = ($localTag.ExitCode -eq 0)
$remoteTag = Invoke-NativeCapture { & git -C $Root ls-remote --exit-code --tags origin "refs/tags/$tag" }
$remoteTagExists = ($remoteTag.ExitCode -eq 0)

if ($tagExists) {
    throw "Tag $tag already exists locally. Bump VERSION and retry."
}
if ($remoteTagExists) {
    throw "Tag $tag already exists on origin. Bump VERSION and retry."
}
Write-Output "[release] tag $tag is available."

# ------------------------------------------------------- 4. frontend build ----
Write-Step 'Building frontend (npm run build)'

Assert-Tool -Name 'node' -Hint 'Install Node.js 22 and retry.'
$npm = Get-NpmCommand
Push-Location $DesktopDir
try {
    if (-not (Test-Path -LiteralPath (Join-Path $DesktopDir 'node_modules') -PathType Container)) {
        Write-Output '[release] node_modules missing, running npm ci'
        Invoke-Native -Command { & $npm ci } -Label 'npm ci'
    }
    Invoke-Native -Command { & $npm run build } -Label 'npm run build'
} finally {
    Pop-Location
}
$distDir = Join-Path $DesktopDir 'dist'
if (-not (Test-Path -LiteralPath $distDir -PathType Container)) {
    throw "Frontend build did not produce $distDir"
}
Write-Output '[release] frontend build OK.'

# ------------------------------------------------------------ 5. backend ----
if ($SkipTests) {
    Write-Output ''
    Write-Output '[release] backend tests skipped (-SkipTests).'
} else {
    Write-Step 'Running backend tests (pytest)'

    $python = Join-Path $Root '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        $python = 'python'
        Assert-Tool -Name 'python' -Hint 'Install Python 3.11 or activate the project virtualenv.'
    }
    Push-Location $BackendDir
    try {
        Invoke-Native -Command { & $python -m pytest -q } -Label 'pytest'
    } finally {
        Pop-Location
    }
    Write-Output '[release] backend tests OK.'
}

# ---------------------------------------------------------------- 6. tag ----
if ($DryRun) {
    Write-Step 'Dry run complete'
    Write-Output "[dry-run] all checks passed; would create annotated tag $tag"
    Write-Output "[dry-run]   git tag -a $tag -m `"LLM-Xray $version`""
    if ($Push) {
        Write-Output "[dry-run]   git push origin $tag"
    } else {
        Write-Output '[dry-run]   (not pushing; pass -Push to push)'
    }
    exit 0
}

Write-Step "Creating annotated tag $tag"

Invoke-Native -Command { & git -C $Root tag -a $tag -m "LLM-Xray $version" } -Label "git tag $tag"
Write-Output "[release] created annotated tag $tag"

if ($Push) {
    Write-Step "Pushing $tag to origin"
    Invoke-Native -Command { & git -C $Root push origin $tag } -Label "git push origin $tag"
    Write-Output "[release] pushed $tag"
} else {
    Write-Output "Next step: git push origin $tag   (or rerun with -Push)"
}

Write-Output ''
Write-Output "[release] released LLM-Xray $version"
exit 0
