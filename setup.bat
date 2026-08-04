@echo off
title Chakra AI - System Setup
cd /d "%~dp0"

echo.
echo =======================================================================
echo  Chakra AI - System Setup
echo  github.com/fir3storm/chakra-ai
echo =======================================================================
echo.
echo  This script will prepare your machine to run Chakra AI.
echo  It installs dependencies and downloads a ~1 GB coding model.
echo.
echo  The 1.56 TB Kimi K3 checkpoint is NOT downloaded (needs explicit flag).
echo =======================================================================
echo.

REM Step 0: Check Python
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo        Found Python %%v

REM Step 1: Install pip dependencies
echo.
echo [2/6] Installing Python dependencies...
pip install --upgrade pip -q
pip install torch transformers huggingface_hub hf_xet psutil -q
echo        Core dependencies installed.

REM Step 2: Install llama-cpp-python (compiles from C++)
echo.
echo [3/6] Installing llama-cpp-python (compiling from C++, takes 2-5 min)...
pip install llama-cpp-python -q
python -c "from llama_cpp import Llama; print('        llama.cpp installed successfully')"

REM Step 3: Install Chakra AI package
echo.
echo [4/6] Installing Chakra AI package...
pip install -e . -q
echo        Package installed.

REM Step 4: Download GGUF model (~1 GB)
echo.
echo [5/6] Checking for local model...

set GGUF_FILE=models\chakra_local\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf
if exist "%GGUF_FILE%" (
    echo        GGUF model already present. Skipping download.
) else (
    echo        Downloading Qwen2.5-Coder-1.5B (Q4_K_M, ~1 GB)...
    python chakra/engine_llama.py --download
    if errorlevel 1 (
        echo [WARN]  GGUF download failed. Falling back to PyTorch safetensors download.
        python tools/download_model.py
    )
)

REM Step 5: Run benchmark
echo.
echo [6/6] Running system benchmark...
python tools/benchmark.py 2>nul || (
    echo        Benchmark skipped (run 'python tools/benchmark.py' manually later)
)

REM Done
echo.
echo =======================================================================
echo  Setup Complete! 
echo.
echo  Launch Chakra AI:
echo    start_chakra_ai.bat
echo.
echo  To download full 1.56 TB Kimi K3 checkpoint (for Engine A/B):
echo    scripts\build_k3.sh          (Linux only)
echo    (See: https://github.com/FareedKhan-dev/kimi-k3-in-c)
echo =======================================================================
echo.
pause
