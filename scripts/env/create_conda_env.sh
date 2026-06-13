#!/usr/bin/env bash
set -euo pipefail # tweaks Bash error handling
env_name="BaggingCPP"

if conda env list | grep -q "$env_name"; then
    echo "Removing existing conda environment '$env_name'..."
    conda env remove -n "$env_name" -y
fi

echo "Creating conda environment 'BaggingCPP' with Python 3.10..."
conda create -n BaggingCPP python=3.10 -y

# Ensure conda is available in this non-interactive shell
echo "Initializing conda for this shell..."
set +u
eval "$(conda shell.bash hook)"


echo "Activating the 'BaggingCPP' environment and installing dependencies..."
conda activate BaggingCPP

echo "Installing mmseqs2 via conda..."
conda install -c bioconda mmseqs2 -y

echo "Installing required packages from requirements.txt..."
pip install -r scripts/env/requirements.txt
set -u

mkdir tmp
mkdir tmp/mmseqs