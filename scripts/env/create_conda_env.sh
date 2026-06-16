#!/usr/bin/env bash
set -euo pipefail # tweaks Bash error handling
env_name="BaggingCPP"

if conda env list | grep -q "$env_name"; then
    echo "Removing existing conda environment '$env_name'..."
    conda env remove -n "$env_name" -y
fi

echo "Creating conda environment '$env_name' with Python 3.10..."
conda create -n "$env_name" python=3.10 -y

# Ensure conda is available in this non-interactive shell
echo "Initializing conda for this shell..."
set +u
eval "$(conda shell.bash hook)"


echo "Activating the '$env_name' environment and installing dependencies..."
conda activate "$env_name"

echo "Installing mmseqs2 via conda..."
conda install -c bioconda -c conda-forge mmseqs2 -y

echo "Installing required packages from requirements.txt..."
pip install -r scripts/env/requirements_lean.txt
set -u

mkdir tmp
mkdir tmp/mmseqs