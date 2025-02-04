#!/bin/bash

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install it first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install pandas matplotlib

# Create requirements.txt
echo "Creating requirements.txt..."
pip freeze > requirements.txt

echo "Virtual environment setup complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To deactivate, run: deactivate"
