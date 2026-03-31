#!/bin/bash
#SBATCH --job-name=fish_image_screener
#SBATCH --ntasks-per-node=5
#SBATCH --partition=GPU
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Load Conda environment
module load anaconda3/2023.09
conda activate fish_instance_segmentation

# Run Fish Image Screener Python program
python scripts/python/fish_image_screener.py --lenient