#!/bin/bash
#SBATCH --job-name=lse-toy-benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=lse.%j.out

# Load conda into this shell session
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Activate the correct Conda environment
conda activate sequence_kernels

# Echo environment diagnostics
echo "Using Python at: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Confirm matplotlib is available in the environment
python -c "import matplotlib; print('[✓] matplotlib found at:', matplotlib.__file__)"

# Run the benchmark script
python /home/s/shuyuan/sequence-kernels/run_benchmark.py
