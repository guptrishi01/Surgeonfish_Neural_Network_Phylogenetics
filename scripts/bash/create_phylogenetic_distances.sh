#!/bin/bash
#SBATCH --job-name=create_distances
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/create_distances_%j.out
#SBATCH --error=logs/create_distances_%j.err

module load cuda/12.8
module load gcc/13.2.0
module load anaconda3/2023.09
conda activate surgeonfish

# Step 1: Building distance matrices
#python scripts/python/build_distance_matrix.py

# Step 2: Running phylogenetic comparison
python scripts/python/compare_to_phylogeny.py --n-permutations 99999
#python scripts/python/build_distance_matrix.py --inspect-tree 2>&1 | head -80