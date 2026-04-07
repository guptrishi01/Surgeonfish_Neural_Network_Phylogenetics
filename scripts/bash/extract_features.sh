#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

module load cuda/12.8
module load gcc/13.2.0
module load anaconda3/2023.09
conda activate surgeonfish

# Extract Features
python scripts/python/extract_features.py
