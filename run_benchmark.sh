#!/bin/bash
#SBATCH --job-name=lse-toy-benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=lse.%j.out
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# 2) Activate your environment
conda activate sequence_kernels
export PYTHONNOUSERSITE=True
echo "Using Python at: $(which python)"
echo "PYTHONNOUSERSITE=$PYTHONNOUSERSITE"
python /home/s/shuyuan/sequence_kernels/run_benchmark.py