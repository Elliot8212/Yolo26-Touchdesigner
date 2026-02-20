@echo off
setlocal

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

set TD_ROOT=%ProgramFiles%\Derivative
set TD_DIR=
for /d %%D in ("%TD_ROOT%\TouchDesigner.*") do set TD_DIR=%%D

if "%TD_DIR%"=="" (
  echo TouchDesigner not found in %TD_ROOT%.
  echo Edit this file and set TD_DIR manually.
  exit /b 1
)

set PY=%TD_DIR%\bin\python.exe
if not exist "%PY%" (
  echo Python not found at %PY%
  echo Edit this file and set TD_DIR manually.
  exit /b 1
)
echo Using TouchDesigner Python: %PY%

set CUDA_TAG=cu118
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  for /f "tokens=3 delims=:" %%A in ('nvidia-smi ^| findstr /C:"CUDA Version"') do set CUDA_VER=%%A
  if not "%CUDA_VER%"=="" (
    for /f "tokens=1 delims= " %%B in ("%CUDA_VER%") do set CUDA_VER=%%B
    for /f "tokens=1 delims=." %%M in ("%CUDA_VER%") do set CUDA_MAJOR=%%M
    for /f "tokens=2 delims=." %%m in ("%CUDA_VER%") do set CUDA_MINOR=%%m
    if "%CUDA_MAJOR%"=="12" (
      set CUDA_TAG=cu121
    ) else if "%CUDA_MAJOR%"=="13" (
      set CUDA_TAG=cu121
    )
  )
)

echo Using PyTorch CUDA wheel: %CUDA_TAG%

"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%CUDA_TAG%
"%PY%" -m pip install -r requirements.txt --prefer-binary

echo Done.
pause
