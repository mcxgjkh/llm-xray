#Requires -Version 5.1
<#
.SYNOPSIS
    Download the optional surrogate lens model for LLM-Xray.

.DESCRIPTION
    The app never downloads models on its own. Run this script explicitly to fetch
    the default lens model (a small causal language model) into the local models
    directory. The lens channel stays disabled until the user enables it in the
    app; nothing is loaded or started here.

    Requires huggingface_hub:
        pip install -r backend/requirements-lens.txt

    Model licenses belong to their publishers. Review the model card before use.

.PARAMETER Model
    Hugging Face repository id. Default: Qwen/Qwen2.5-0.5B-Instruct

.PARAMETER Destination
    Local target directory. Default: models/lens/<sanitized-model-name>

.PARAMETER Endpoint
    Hugging Face endpoint. Defaults to $env:HF_ENDPOINT when set, otherwise the
    official host. If the official host is unreachable, the script retries once
    through https://hf-mirror.com (common mirror for mainland China).

.PARAMETER DryRun
    Print what would be downloaded without touching the network.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/fetch_lens_model.ps1 -DryRun

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/fetch_lens_model.ps1

.EXAMPLE
    $env:HF_ENDPOINT = 'https://hf-mirror.com'; powershell -File scripts/fetch_lens_model.ps1
#>
[CmdletBinding()]
param(
    [string]$Model = 'Qwen/Qwen2.5-0.5B-Instruct',
    [string]$Destination,
    [string]$Endpoint = $env:HF_ENDPOINT,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$Root = (Resolve-Path -LiteralPath $Root).Path

if (-not $Destination) {
    $safeName = $Model -replace '[/\\:]', '__'
    $Destination = Join-Path $Root (Join-Path 'models\lens' $safeName)
}
$Destination = [System.IO.Path]::GetFullPath($Destination)

$python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw 'Python 3.11 was not found. Create the virtualenv described in README.md.'
    }
    $python = 'python'
}

Write-Output "[lens-model] model:       $Model"
Write-Output "[lens-model] destination: $Destination"
Write-Output ''

if ($DryRun) {
    Write-Output '[dry-run] would run: python -c "from huggingface_hub import snapshot_download; ..."'
    Write-Output "[dry-run] download source: https://huggingface.co/$Model"
    Write-Output "[dry-run] target directory: $Destination"
    Write-Output '[dry-run] no network access was made.'
    exit 0
}

# --------------------------------------------------------- dependency check ----
$previousErrorAction = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python -c 'import huggingface_hub' 2>$null | Out-Null
$hubAvailable = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $previousErrorAction

if (-not $hubAvailable) {
    Write-Output "[lens-model] huggingface_hub is unavailable for '$python'; using the ModelScope fallback."
}

New-Item -ItemType Directory -Path $Destination -Force | Out-Null

# --------------------------------------------------------------- download ----
$downloadCode = @'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
target = sys.argv[2]
path = snapshot_download(repo_id=repo_id, local_dir=target)
print(path)
'@

Write-Output '[lens-model] downloading (this can take several minutes)...'
$previousErrorAction = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$downloadExit = 1
$attempts = @()
if ($Endpoint) {
    $attempts = @($Endpoint)
} else {
    $attempts = @('https://huggingface.co', 'https://hf-mirror.com')
}
try {
    if ($hubAvailable) {
        foreach ($candidate in $attempts) {
            $env:HF_ENDPOINT = $candidate
            Write-Output "[lens-model] endpoint: $candidate"
            & $python -c $downloadCode $Model $Destination
            $downloadExit = $LASTEXITCODE
            if ($downloadExit -eq 0) { break }
            Write-Output "[lens-model] endpoint failed; trying the next source if any"
        }
    }
} finally {
    $ErrorActionPreference = $previousErrorAction
}

# Many networks in mainland China can reach ModelScope but not the HF CDN.
# Fall back to ModelScope's direct file API (curl.exe is part of Windows 10+).
if ($downloadExit -ne 0 -and $Model -match '^[^/]+/[^/]+$' -and (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    Write-Output '[lens-model] trying ModelScope fallback'
    $msApi = "https://modelscope.cn/api/v1/models/$Model/repo/files?Revision=master"
    $msBase = "https://modelscope.cn/models/$Model/resolve/master"
    try {
        $listing = Invoke-RestMethod -Uri $msApi -TimeoutSec 30
        $files = @($listing.Data.Files | Where-Object { $_.Type -eq 'blob' })
        if ($files.Count -eq 0) { throw 'ModelScope returned no files' }
        $msFailed = 0
        foreach ($file in $files) {
            $targetFile = Join-Path $Destination $file.Name
            $parent = Split-Path -Parent $targetFile
            if ($parent -and -not (Test-Path -LiteralPath $parent)) {
                New-Item -ItemType Directory -Force -Path $parent | Out-Null
            }
            if ((Test-Path -LiteralPath $targetFile) -and ((Get-Item -LiteralPath $targetFile).Length -eq [int64]$file.Size)) {
                Write-Output "[lens-model] skip complete $($file.Name)"
                continue
            }
            Write-Output "[lens-model] modelscope: $($file.Name) ($([math]::Round($file.Size / 1MB, 1)) MB)"
            & curl.exe -L --fail --retry 3 --retry-delay 2 -C - -o $targetFile "$msBase/$($file.Name)"
            if ($LASTEXITCODE -ne 0) { $msFailed++; Write-Output "[lens-model] failed $($file.Name)" }
        }
        if ($msFailed -eq 0) { $downloadExit = 0 }
    } catch {
        Write-Output "[lens-model] ModelScope fallback failed: $($_.Exception.Message)"
    }
}

if ($downloadExit -ne 0) {
    throw "Model download failed. Check the network, or download manually from ModelScope/Hugging Face into $Destination."
}

Write-Output ''
Write-Output "[lens-model] done: $Destination"
Write-Output "[lens-model] local path: $Destination"
Write-Output '[lens-model] the app resolves models/lens automatically. Alternatively set:'
Write-Output "    `$env:LLM_XRAY_LENS_MODEL = '$Destination'"
Write-Output '[lens-model] next: install backend/requirements-lens.txt, start the app, then enable the lens channel.'
Write-Output '[lens-model] the app will not load the model until you enable the lens channel.'
