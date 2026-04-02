#!/bin/bash
#SBATCH --job-name=generate_annotations
#SBATCH --ntasks-per-node=8
#SBATCH --partition=GPU
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err


# Run generate_annotations.py
python scripts/python/generate_annotations.py

# Command to run slurm script
# sbatch scripts/bash/generate_annotations.sh
