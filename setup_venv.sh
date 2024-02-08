#!/bin/bash

# Set up Python version
python_version="3.9.7"

# Set up virtual environment
venv_name="venv"

if [ -n "$1" ]; then
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu"${1}"
else
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Check if virtualenv is installed
if ! command -v virtualenv &> /dev/null; then
    echo "virtualenv is not installed. Installing..."
    apt install python3.8-venv
    pip install virtualenv
fi

# Create and activate virtual environment
python3 -m venv "$venv_name"
source "$venv_name/bin/activate"

# Install Python dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt


# Deactivate virtual environment
# deactivate

echo "Virtual environment '$venv_name' has been set up with Python $python_version and dependencies installed."
