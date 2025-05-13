#!/bin/bash

# Name of the virtual environment folder
VENV_DIR=".venv"

# Check if the virtual environment already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists."
else
    echo "Creating a virtual environment..."

    # Create the virtual environment
    python3.11 -m venv $VENV_DIR
    echo "Virtual environment created."
fi

# Activate the virtual environment
echo "Activating the virtual environment..."
source $VENV_DIR/bin/activate

# Optional: Install dependencies (uncomment if you have a requirements.txt)
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Done
echo "Virtual environment is ready!"
