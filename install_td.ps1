#Requires -Version 5.1
<#
.SYNOPSIS
  Installs PyTorch + YOLO26 dependencies into TouchDesigner's embedded Python.

.DESCRIPTION
  - Locates TouchDesigner Python (handles both versioned and non-versioned install paths).
  - Detects the NVIDIA GPU compute capability and picks the matching PyTorch CUDA wheel.
    * RTX 50XX (Blackwell, sm_12.0)  -> cu128
    * RTX 30XX / 40XX (sm_8.x / 9.x) -> cu124
    * RTX 20XX / 16XX  (sm_7.x)      -> cu121
    * GTX 10XX et plus ancien        -> cu118
    * Pas de GPU NVIDIA              -> cpu
  - Installe torch, torchvision, torchaudio puis requirements.txt.
  - Vérifie l'installation à la fin.

.NOTES
  Les paquets s'installent dans le user site-packages (%APPDATA%\Python\Python311\site-packages)
  car bin\ de TouchDesigner est read-only sans droits admin. C'est normal et fonctionne.
#>

$ErrorActionPreference = "Stop"

function Write-Step($n, $total, $msg) {
    Write-Host ""
    Write-Host "[$n/$total] $msg" -ForegroundColor Cyan
}

function Write-Ok($msg) {
    Write-Host "  -> $msg" -ForegroundColor Green
}

function Write-Warn($msg) {
    Write-Host "  ! $msg" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host " YOLO26 + TouchDesigner installer" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

# -------------------------------------------------------------------
Write-Step 1 5 "Recherche de TouchDesigner Python..."
$tdRoot = Join-Path $env:ProgramFiles "Derivative"

if (-not (Test-Path $tdRoot)) {
    Write-Error "$tdRoot n'existe pas. TouchDesigner est-il installe ?"
    exit 1
}

$candidates = Get-ChildItem $tdRoot -Directory -Filter "TouchDesigner*" -ErrorAction SilentlyContinue |
    ForEach-Object { Join-Path $_.FullName "bin\python.exe" } |
    Where-Object { Test-Path $_ } |
    Sort-Object -Unique

if (-not $candidates -or $candidates.Count -eq 0) {
    Write-Error "TouchDesigner Python introuvable dans $tdRoot."
    exit 1
}

$TD_PY = $candidates | Sort-Object { (Get-Item $_).LastWriteTime } -Descending | Select-Object -First 1
Write-Ok $TD_PY

$pyVer = (& $TD_PY --version 2>&1) -join " "
Write-Ok $pyVer

# -------------------------------------------------------------------
Write-Step 2 5 "Detection de la GPU..."
$cudaTag = "cpu"
$gpuName = "(aucune)"
$cc = $null

$nvSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvSmi) {
    try {
        $cc = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
        $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    } catch {}

    if ([string]::IsNullOrWhiteSpace($cc)) {
        # nvidia-smi sans champ compute_cap (driver tres ancien) -> fallback prudent
        Write-Warn "compute_cap indisponible, fallback cu121"
        $cudaTag = "cu121"
    } else {
        Write-Ok "GPU : $gpuName"
        Write-Ok "Compute capability : $cc"

        if ($cc -match "^(\d+)\.(\d+)$") {
            $major = [int]$Matches[1]
            if     ($major -ge 12) { $cudaTag = "cu128" }   # Blackwell (RTX 50XX, B100)
            elseif ($major -ge 9)  { $cudaTag = "cu126" }   # Hopper (H100)
            elseif ($major -ge 8)  { $cudaTag = "cu124" }   # Ampere (30XX, A100), Ada (40XX)
            elseif ($major -ge 7)  { $cudaTag = "cu121" }   # Volta (V100), Turing (20XX, 16XX)
            else                    { $cudaTag = "cu118" }  # Pascal (10XX) et plus anciens
        }
    }
} else {
    Write-Warn "nvidia-smi introuvable - installation CPU-only"
}

Write-Ok "Wheel PyTorch : $cudaTag"

# -------------------------------------------------------------------
Write-Step 3 5 "Mise a jour de pip..."
& $TD_PY -m pip install --upgrade pip --quiet --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { Write-Warn "Echec upgrade pip (non-bloquant)" }

# -------------------------------------------------------------------
Write-Step 4 5 "Installation de PyTorch ($cudaTag) - peut prendre quelques minutes..."

$indexUrl = if ($cudaTag -eq "cpu") {
    "https://download.pytorch.org/whl/cpu"
} else {
    "https://download.pytorch.org/whl/$cudaTag"
}

& $TD_PY -m pip install --upgrade --force-reinstall `
    torch torchvision torchaudio `
    --index-url $indexUrl `
    --disable-pip-version-check

if ($LASTEXITCODE -ne 0) {
    Write-Error "L'installation de PyTorch a echoue. Verifie ta connexion et reessaie."
    exit 1
}

# -------------------------------------------------------------------
Write-Step 5 5 "Installation des dependances YOLO (ultralytics, opencv, lapx)..."
$reqFile = Join-Path $PSScriptRoot "requirements.txt"
if (Test-Path $reqFile) {
    & $TD_PY -m pip install -r $reqFile --prefer-binary --disable-pip-version-check
} else {
    Write-Warn "requirements.txt introuvable, fallback liste manuelle"
    & $TD_PY -m pip install ultralytics opencv-python numpy lapx --prefer-binary --disable-pip-version-check
}

if ($LASTEXITCODE -ne 0) {
    Write-Warn "Au moins un paquet n'a pas pu s'installer. Continue quand meme la verification."
}

# -------------------------------------------------------------------
Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host " Verification" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

$verifyCode = @'
import sys, importlib

def show(label, fn):
    try:
        print(f"  {label:<14}: {fn()}")
    except Exception as e:
        print(f"  {label:<14}: ERREUR - {e}")

show("Python", lambda: sys.version.split()[0])
show("torch", lambda: importlib.import_module("torch").__version__)

import torch
show("CUDA dispo", lambda: torch.cuda.is_available())
if torch.cuda.is_available():
    show("CUDA build", lambda: torch.version.cuda)
    show("GPU", lambda: torch.cuda.get_device_name(0))
show("torchvision", lambda: importlib.import_module("torchvision").__version__)
show("ultralytics", lambda: importlib.import_module("ultralytics").__version__)
show("opencv", lambda: importlib.import_module("cv2").__version__)
show("numpy", lambda: importlib.import_module("numpy").__version__)
try:
    import lap
    print(f"  {'lap (lapx)':<14}: {lap.__version__}")
except Exception as e:
    print(f"  {'lap (lapx)':<14}: NON installe ({e}) - tracking ByteTrack/BoT-SORT desactive")
'@

& $TD_PY -c $verifyCode

Write-Host ""
Write-Host "===========================================" -ForegroundColor Green
Write-Host " Installation terminee" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Prochaines etapes :"
Write-Host "  1. Ouvre Yolo26.toe dans TouchDesigner"
Write-Host "  2. Ou charge td_yolo26.py dans un Text DAT + Script TOP"
Write-Host ""
