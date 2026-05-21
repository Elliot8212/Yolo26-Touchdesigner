@echo off
REM Launcher pour install_td.ps1 - permet le double-clic depuis l'Explorateur Windows
REM sans toucher a la policy d'execution PowerShell de la machine.

setlocal
set "SCRIPT_DIR=%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install_td.ps1"
set "RC=%ERRORLEVEL%"

echo.
pause
exit /b %RC%
