#!/bin/bash
#SBATCH --job-name=standardize_images
#SBATCH --ntasks-per-node=8
#SBATCH --partition=GPU
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err


# Run standardize_images.py
python scripts/python/standardize_images.py

# Command to run slurm script
# sbatch scripts/bash/standardize_images.sh
