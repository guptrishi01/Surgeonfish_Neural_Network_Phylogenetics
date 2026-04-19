#!/bin/bash
#SBATCH --job-name=run_scripts
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/run_scripts_%j.out
#SBATCH --error=logs/run_scripts_%j.err

module load cuda/12.8
module load gcc/13.2.0
module load anaconda3/2023.09
conda activate surgeonfish

# Run standardize_images.py
python scripts/python/standardize_images.py

# Run generate_annotations.py
python scripts/python/generate_annotations.py

# Run prepare_splits.py
python scripts/python/prepare_splits.py

# Iteration 1: train then immediately inspect val
#python scripts/python/train_mask_rcnn.py --mode train
#python scripts/python/train_mask_rcnn.py --mode val

# Once val masks look correct, run test ONCE:
python scripts/python/train_mask_rcnn.py --mode test

# Evaluate and ensure creation of model
sbatch scripts/bash/evaluate_model.sh

# Extract Features
python scripts/python/extract_features.py

# Extract Features
python scripts/python/validate_features.py --report

# Visualize full dataset (takes ~15 min, one figure per species)
python scripts/python/visualize_features.py

# Build distance matrices
#python scripts/python/build_distance_matrix.py

# Run phylogenetic comparison
python scripts/python/compare_to_phylogeny.py --n-permutations 99999