#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --ntasks-per-node=8
#SBATCH --partition=GPU
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err


# Load modules
module load cuda/12.4
module load gcc/13.2.0
module load anaconda3/2023.09

# Create Conda environment
conda create -n surgeonfish python=3.10 -y
source activate surgeonfish

# PyTorch -- edit cuda version to match your cluster
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Core image processing
conda install -c conda-forge opencv numpy pillow -y

# Pip packages
pip install sam2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu124/torch2.3/index.html
pip install matplotlib seaborn pandas scipy scikit-learn pycocotools ultralytics

#echo "Environment setup complete."
python -c "import torch; print('CUDA:', torch.cuda.is_available())"