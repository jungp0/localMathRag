@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-LocalMathRAG.ps1" %*
endlocal
