#!/bin/bash
# Chakra AI - System Setup
# github.com/fir3storm/chakra-ai
# This script prepares your machine to run Chakra AI.
# The 1.56 TB Kimi K3 checkpoint is NOT downloaded unless explicitly told.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "======================================================================="
echo "  Chakra AI - System Setup"
echo "  github.com/fir3storm/chakra-ai"
echo "======================================================================="
echo ""
echo "  This script will prepare your machine to run Chakra AI."
echo "  It installs dependencies and downloads a ~1 GB coding model."
echo ""
echo "  The 1.56 TB Kimi K3 checkpoint is NOT downloaded (needs --with-k3 flag)."
echo "======================================================================="
echo ""

# Step 0: Check Python
echo "[1/6] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Install Python 3.10+"
    exit 1
fi
echo "       Found $(python3 --version)"

# Step 1: Install pip dependencies
echo ""
echo "[2/6] Installing Python dependencies..."
pip3 install --upgrade pip -q
pip3 install torch transformers huggingface_hub hf_xet psutil -q
echo "       Core dependencies installed."

# Step 2: Install llama-cpp-python
echo ""
echo "[3/6] Installing llama-cpp-python (compiling from C++, takes 2-5 min)..."
CMAKE_ARGS="-DLLAMA_AVX2=ON" pip3 install llama-cpp-python -q
python3 -c "from llama_cpp import Llama; print('       llama.cpp installed successfully')"

# Step 3: Install Chakra AI package
echo ""
echo "[4/6] Installing Chakra AI package..."
pip3 install -e . -q
echo "       Package installed."

# Step 4: Download GGUF model (~1 GB)
echo ""
echo "[5/6] Checking for local model..."
GGUF_FILE="models/chakra_local/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"
if [ -f "$GGUF_FILE" ]; then
    echo "       GGUF model already present. Skipping download."
else
    echo "       Downloading Qwen2.5-Coder-1.5B (Q4_K_M, ~1 GB)..."
    python3 chakra/engine_llama.py --download || {
        echo "[WARN]  GGUF download failed. Falling back to PyTorch safetensors download."
        python3 tools/download_model.py
    }
fi

# Step 5: Run benchmark
echo ""
echo "[6/6] Running system benchmark..."
python3 tools/benchmark.py 2>/dev/null || echo "       Benchmark skipped (run 'python3 tools/benchmark.py' manually later)"

# Optional: Build kimi-k3-in-c for Engine A
if [ "$1" = "--with-k3" ]; then
    echo ""
    echo "[7/7] Building kimi-k3-in-c for Engine A..."
    bash scripts/build_k3.sh
fi

# Done
echo ""
echo "======================================================================="
echo "  Setup Complete!"
echo ""
echo "  Launch Chakra AI:"
echo "    python3 -m chakra.cli"
echo ""
echo "  To build kimi-k3-in-c (Engine A, needs 1.56 TB checkpoint):"
echo "    bash setup.sh --with-k3"
echo "======================================================================="
echo ""
