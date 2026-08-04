# Chakra AI - One-Line Installer
# Run in PowerShell as Administrator:
#
#   powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/fir3storm/chakra-ai/main/install.ps1 | iex"
#
# After install, just type: chakra

$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/fir3storm/chakra-ai.git"
$InstallDir = "$env:USERPROFILE\chakra-ai"

Write-Host ""
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "  Chakra AI - One-Line Installer" -ForegroundColor Cyan
Write-Host "  github.com/fir3storm/chakra-ai" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check prerequisites
Write-Host "[1/5] Checking prerequisites..." -ForegroundColor White
$python = Get-Command python -ErrorAction SilentlyContinue
$git = Get-Command git -ErrorAction SilentlyContinue

if (-not $python) {
    Write-Host "[ERROR] Python 3.10+ required. Install from https://python.org" -ForegroundColor Red
    exit 1
}
if (-not $git) {
    Write-Host "[ERROR] Git required. Install from https://git-scm.com" -ForegroundColor Red
    exit 1
}
Write-Host "       Python: $($python.Source)" -ForegroundColor Gray
Write-Host "       Git:    $($git.Source)" -ForegroundColor Gray

# Step 2: Clone repo
Write-Host "[2/5] Cloning Chakra AI..." -ForegroundColor White
if (Test-Path $InstallDir) {
    Write-Host "       Already cloned. Updating..." -ForegroundColor Gray
    Set-Location $InstallDir
    git pull -q
} else {
    git clone -q $RepoUrl $InstallDir
    Set-Location $InstallDir
}
Write-Host "       Repository ready." -ForegroundColor Gray

# Step 3: Install Python dependencies
Write-Host "[3/5] Installing Python dependencies..." -ForegroundColor White
python -m pip install --upgrade pip -q
pip install torch transformers huggingface_hub hf_xet psutil -q
Write-Host "       Core deps installed." -ForegroundColor Gray

# Step 4: Compile fast engine + install package
Write-Host "[4/5] Compiling llama.cpp engine (takes 2-5 min)..." -ForegroundColor White
pip install llama-cpp-python -q
pip install -e . -q
Write-Host "       Fast engine compiled." -ForegroundColor Gray

# Step 5: Download model (~1 GB)
Write-Host "[5/5] Checking for model..." -ForegroundColor White
$ggufFile = "$InstallDir\models\chakra_local\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"
if (Test-Path $ggufFile) {
    Write-Host "       Model already present. Skipping download." -ForegroundColor Gray
} else {
    Write-Host "       Downloading Qwen2.5-Coder-1.5B (Q4_K_M, ~1 GB)..." -ForegroundColor White
    python chakra/engine_llama.py --download
}

# Create chakra.bat in PATH
$batchFile = "$env:USERPROFILE\chakra.bat"
@"
@echo off
cd /d "$InstallDir"
python -m chakra.cli %*
"@ | Out-File -FilePath $batchFile -Encoding ASCII

# Create chakra.ps1 in PATH
$ps1File = "$env:USERPROFILE\chakra.ps1"
@"
Set-Location "$InstallDir"
python -m chakra.cli @args
"@ | Out-File -FilePath $ps1File -Encoding ASCII

# Done
Write-Host ""
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "  Chakra AI installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  To start, open a NEW terminal and type:" -ForegroundColor White
Write-Host "    chakra" -ForegroundColor Yellow
Write-Host ""
Write-Host "  The 1.56 TB Kimi K3 checkpoint was NOT downloaded." -ForegroundColor Gray
Write-Host "  Install location: $InstallDir" -ForegroundColor Gray
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host ""
