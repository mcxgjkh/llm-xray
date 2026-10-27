#Requires -Version 5.1
<#
.SYNOPSIS
    Verify that the project version is consistent across all versioned files.

.DESCRIPTION
    Checks VERSION, apps/desktop/package.json, CHANGELOG.md and docs/VERSIONS.md.
    Exits 0 when every check passes, 1 when any file is missing, unreadable or
    inconsistent. Used by CI and by scripts/release.ps1.

.PARAMETER Root
    Repository root. Defaults to the parent directory of this script.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/check-version.ps1
#>
[CmdletBinding()]
param(
    [string]$Root
)

$ErrorActionPreference = 'Stop'

if (-not $Root) {
    $Root = Split-Path -Parent $PSScriptRoot
}
$Root = (Resolve-Path -LiteralPath $Root).Path

$script:Failures = New-Object 'System.Collections.Generic.List[string]'
$script:Checks = 0

function Add-CheckResult {
    param(
        [bool]$Ok,
        [string]$Label,
        [string]$Detail
    )
    $script:Checks++
    if ($Ok) {
        Write-Output ("[ok]   {0}{1}" -f $Label, $(if ($Detail) { " = $Detail" } else { '' }))
    } else {
        Write-Output ("[FAIL] {0}{1}" -f $Label, $(if ($Detail) { ": $Detail" } else { '' }))
        $script:Failures.Add($Label)
    }
}

function Read-Utf8Text {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8)
}

Write-Output "[check-version] repository: $Root"

# ---------------------------------------------------------------- VERSION ----
$versionFile = Join-Path $Root 'VERSION'
$versionText = Read-Utf8Text -Path $versionFile
$version = if ($null -ne $versionText) { $versionText.Trim() } else { '' }

$versionFormatOk = ($version -match '^\d+\.\d+\.\d+$')
Add-CheckResult -Ok $versionFormatOk -Label 'VERSION exists and is semver' -Detail $(if ($version) { $version } else { 'missing or empty' })

# ------------------------------------------------------------ package.json ----
$packageFile = Join-Path $Root 'apps\desktop\package.json'
$packageVersion = ''
$packageOk = $false
$packageText = Read-Utf8Text -Path $packageFile
if ($null -ne $packageText) {
    try {
        $packageJson = $packageText | ConvertFrom-Json
        $packageVersion = [string]$packageJson.version
        $packageOk = [bool]($packageVersion -match '^\d+\.\d+\.\d+$') -and ($packageVersion -eq $version)
    } catch {
        $packageOk = $false
        $packageVersion = "invalid JSON: $($_.Exception.Message)"
    }
} else {
    $packageVersion = 'missing file: apps/desktop/package.json'
}
Add-CheckResult -Ok $packageOk -Label 'apps/desktop/package.json version matches VERSION' -Detail $packageVersion

# -------------------------------------------------------------- CHANGELOG ----
$changelogFile = Join-Path $Root 'CHANGELOG.md'
$changelogText = Read-Utf8Text -Path $changelogFile
$changelogOk = $false
$changelogDetail = 'missing file: CHANGELOG.md'
if ($null -ne $changelogText -and $version) {
    $pattern = '##\s+\[' + [regex]::Escape($version) + '\]\s+-\s+\d{4}-\d{2}-\d{2}'
    $match = [regex]::Match($changelogText, $pattern)
    if ($match.Success) {
        $changelogOk = $true
        $changelogDetail = $match.Value
    } else {
        $changelogDetail = "missing heading '## [$version] - YYYY-MM-DD'"
    }
}
Add-CheckResult -Ok $changelogOk -Label 'CHANGELOG.md has a dated entry for VERSION' -Detail $changelogDetail

# ----------------------------------------------------------- docs/VERSIONS ----
$versionsFile = Join-Path $Root 'docs\VERSIONS.md'
$versionsText = Read-Utf8Text -Path $versionsFile
$versionsOk = $false
$versionsDetail = 'missing file: docs/VERSIONS.md'
if ($null -ne $versionsText -and $version) {
    $rowPattern = '\|\s*' + [regex]::Escape($version) + '\s*\|'
    if ([regex]::IsMatch($versionsText, $rowPattern)) {
        $versionsOk = $true
        $versionsDetail = "version row found"
    } else {
        $versionsDetail = "missing table row for $version"
    }
}
Add-CheckResult -Ok $versionsOk -Label 'docs/VERSIONS.md version table lists VERSION' -Detail $versionsDetail

Write-Output ''
Write-Output "[check-version] $($script:Checks) check(s), $($script:Failures.Count) failure(s)"

if ($script:Failures.Count -gt 0) {
    Write-Output "Version consistency check failed. Fix the files above and retry."
    exit 1
}

Write-Output "Version consistency check passed: $version"
exit 0
