#!/bin/bash
#SBATCH --job-name=surgeonfish_train
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module load cuda/12.8
module load gcc/13.2.0
module load anaconda3/2023.09
conda activate surgeonfish

# Iteration 1: train then immediately inspect val
#python scripts/python/train_mask_rcnn.py --mode train
#python scripts/python/train_mask_rcnn.py --mode val

# After inspecting val_predictions/ overlays:

# If annotation is bad (SAM drew the wrong region):
#   Fix annotations.json manually, then re-run splits
#python scripts/python/prepare_splits.py
#sbatch run_pipeline.sh   # retrain from scratch

# For a single annotation, run fix_annotation.py on image
#python scripts/python/fix_annotation.py --species "Prionurus chrysurus"

# Retrain from scratch
#python scripts/python/train_mask_rcnn.py --mode train
#python scripts/python/train_mask_rcnn.py --mode train \
#    --resume outputs/checkpoints/best_model.pth \
#    --epochs 50
#python scripts/python/train_mask_rcnn.py \
#    --mode train \
#    --unfreeze-backbone \
#    --epochs 50
#python scripts/python/train_mask_rcnn.py --mode val

# If prediction is bad but annotation is correct:
#   Just resume training for more epochs
#python scripts/python/train_mask_rcnn.py --mode train \
#    --resume outputs/checkpoints/best_model.pth --epochs 30

python scripts/python/train_mask_rcnn.py --mode val

# Once val masks look correct, run test ONCE:
python scripts/python/train_mask_rcnn.py --mode test

# Predict mask for Naso annulatus (no SAM annotation):
python scripts/python/train_mask_rcnn.py --mode predict \
    --image "data/standardized_images/Naso/Naso annulatus.png"
