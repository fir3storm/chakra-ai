#!/bin/bash
# Build kimi-k3-in-c for use as Chakra-AI Engine A backend.
# Requires: Linux x86-64, GCC >= 9 or Clang >= 10, make, OpenMP
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
K3_DIR="$PROJECT_DIR/third_party/kimi-k3-in-c"
K3_BIN="$PROJECT_DIR/third_party/k3/bin/k3"

echo "=== Chakra-AI Engine A: kimi-k3-in-c Build ==="
echo ""

# Step 1: Clone if not present
if [ ! -d "$K3_DIR" ]; then
    echo "[1/4] Cloning kimi-k3-in-c..."
    mkdir -p "$PROJECT_DIR/third_party"
    git clone https://github.com/FareedKhan-dev/kimi-k3-in-c.git "$K3_DIR"
else
    echo "[1/4] kimi-k3-in-c already cloned."
fi

# Step 2: Build
echo "[2/4] Building k3 binary..."
cd "$K3_DIR"
make -j

# Step 3: Verify (weightless tests, no model needed)
echo "[3/4] Running verification tests..."
make test

# Step 4: Copy binary
echo "[4/4] Installing k3 binary..."
mkdir -p "$(dirname "$K3_BIN")"
cp "$K3_DIR/bin/k3" "$K3_BIN"
chmod +x "$K3_BIN"

echo ""
echo "=== Build Complete ==="
echo "Binary: $K3_BIN"
echo ""
echo "Next steps:"
echo "  1. Download the Kimi K3 checkpoint (1.56 TB):"
echo "     export HF_TOKEN=hf_your_token"
echo "     $K3_DIR/scripts/download-model.sh ~/k3model"
echo ""
echo "  2. Pack the trunk (one-time, ~4 min):"
echo "     $K3_DIR/scripts/pack-trunk.sh ~/k3model ~/k3trunk"
echo ""
echo "  3. Run Chakra-AI with Engine A:"
echo "     python -m kimipy.cli --trunk ~/k3trunk --engine c-backend"
