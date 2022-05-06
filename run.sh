#!/bin/zsh

# Create the virtual environment
echo "  0. Creating the virtual environment"
python -m venv minecraft_ai_fisher

# Acticate the python virtual environment
echo "	1. Activating the virtual environment"
source minecraft_ai_fisher/bin/activate

echo "  1.1 Upgrading pip in the virtual environment"
pip install pip --upgrade

# Install requirements within the environment
echo "	2. Installing requirements.txt"
python -m pip install -r requirements.txt

# Run
python main.py