#!/bin/bash
# Install bitsandbytes with proper CUDA support

set -e

echo "=================================="
echo "Installing bitsandbytes for CUDA"
echo "=================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSHOP_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if it exists
if [ -d "$WORKSHOP_DIR/.venv" ]; then
    echo "✓ Found virtual environment at $WORKSHOP_DIR/.venv"
    source "$WORKSHOP_DIR/.venv/bin/activate"
elif [ -d "$WORKSHOP_DIR/venv" ]; then
    echo "✓ Found virtual environment at $WORKSHOP_DIR/venv"
    source "$WORKSHOP_DIR/venv/bin/activate"
else
    echo "⚠ No virtual environment found. Using system Python."
fi

echo ""
echo "Python: $(which python3)"
echo "Pip: $(which pip)"
echo ""

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
else
    echo "⚠ CUDA/nvidia-smi not found. Installing CPU version."
fi

echo ""
echo "Installing/upgrading bitsandbytes..."
pip install --upgrade pip
pip install --upgrade bitsandbytes

echo ""
echo "Verifying installation..."
python3 -c "import bitsandbytes as bnb; print(f'✓ bitsandbytes {bnb.__version__} installed at {bnb.__file__}')"

echo ""
echo "=================================="
echo "✓ Installation complete!"
echo "=================================="
echo ""
echo "Now:"
echo "1. Restart your Jupyter kernel"
echo "2. Run the first cell to verify"

