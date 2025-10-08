#!/bin/bash
# setup_env.sh - Script to set up Python virtual environment for the project

echo "Setting up Python virtual environment for Attention CUDA project..."

# Check if Python 3.12 is installed
if command -v python3.12 &>/dev/null; then
    PYTHON_CMD=python3.12
elif command -v python3 &>/dev/null; then
    # Check Python version
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$PY_VERSION" == "3.12" ]]; then
        PYTHON_CMD=python3
    else
        echo "Error: Python 3.12 is not found. Please install Python 3.12 and try again."
        exit 1
    fi
else
    echo "Error: Python 3 is not found. Please install Python 3.12 and try again."
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please make sure venv module is available."
        exit 1
    fi
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

echo
echo "Virtual environment setup complete!"
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo
echo "To deactivate the virtual environment, run:"
echo "  deactivate"
