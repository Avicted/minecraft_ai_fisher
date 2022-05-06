#!/bin/zsh

# Acticate the python virtual environment
echo "	1. Activating the virtual environment"
source minecraft_ai_fisher/bin/activate

# Install requirements within the environment
echo "	2. Installing requirements.txt"
python -m pip install -r requirements.txt

# Start jupyter notebook from within this activated environment
echo "	3. Starting jupyter notebook"
python -m jupyter notebook
