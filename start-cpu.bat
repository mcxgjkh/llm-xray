@echo off
cd /d %~dp0
set XRAY_NATIVE_DEVICE=cpu
python desktop\launcher.py
