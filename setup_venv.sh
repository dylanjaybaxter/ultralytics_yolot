#!/bin/bash

# Set up Python version
python_version="3.9.7"

# Set up virtual environment
venv_name="venv"

# Check if virtualenv is installed
if ! command -v virtualenv &> /dev/null; then
    echo "virtualenv is not installed. Installing..."
    apt install python3.8-venv
    python3 -m pip install virtualenv
fi

# Create and activate virtual environment
python3 -m venv "$venv_name"
source "./$venv_name/bin/activate"

if [ -n "$1" ]; then
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu"${1}"
else
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Install Python dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Deactivate virtual environment
# deactivate

echo "Virtual environment '$venv_name' has been set up with Python $python_version and dependencies installed."
