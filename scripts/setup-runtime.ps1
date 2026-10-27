#Requires -Version 5.1
<#
.SYNOPSIS
  Install the CUDA or CPU PyTorch runtime for LLM-Xray native mode.
.DESCRIPTION
  The application has one build; native mode selects CPU or CUDA in
  Settings -> 原生精确模型 -> 设备. This script installs the matching wheel.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup-runtime.ps1 -Device cuda
#>
[CmdletBinding()]
param([ValidateSet('cuda','cpu')][string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = 'python' }
if ($Device -eq 'cuda') {
  Write-Host '[runtime] installing CUDA PyTorch (torch 2.11.0+cu128)...'
  & $python -m pip install --upgrade --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
} else {
  Write-Host '[runtime] installing CPU PyTorch...'
  & $python -m pip install --upgrade --no-cache-dir torch==2.11.0
}
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed ($LASTEXITCODE)" }
& $python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
