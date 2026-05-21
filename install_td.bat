@echo off
REM Launcher for install_td.ps1 - allows double-clicking from Windows Explorer
REM without changing the machine's PowerShell execution policy.

setlocal
set "SCRIPT_DIR=%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install_td.ps1"
set "RC=%ERRORLEVEL%"

echo.
pause
exit /b %RC%
