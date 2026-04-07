#!/bin/bash
# ==============================================================================
# SLURM batch script template -- surgeonfish pipeline
#
# Usage:
#   sbatch slurm/run_script.sh scripts/python/evaluate_model.py
#   sbatch slurm/run_script.sh scripts/python/train_mask_rcnn.py --mode val
#   sbatch slurm/run_script.sh scripts/python/train_mask_rcnn.py --mode train
#
# If no argument is given, defaults to evaluate_model.py
# ==============================================================================

#SBATCH --job-name=surgeonfish
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------

module load cuda/12.8
module load gcc/13.2.0
module load anaconda3/2023.09
conda activate surgeonfish

# ------------------------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------------------------

echo "========================================"
echo "Job:        $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node:       $SLURMD_NODENAME"
echo "Partition:  $SLURM_JOB_PARTITION"
echo "Started:    $(date)"
echo "========================================"

# Confirm GPU is visible
python3 -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:      {torch.cuda.get_device_name(0)}')
    print(f'VRAM:     {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB')
"

# Ensure log directory exists
mkdir -p logs

# ------------------------------------------------------------------------------
# Run
# Default to evaluate_model.py if no argument supplied
# ------------------------------------------------------------------------------

SCRIPT=${1:-scripts/python/evaluate_model.py}
ARGS="${@:2}"

echo ""
echo "Running: python3 $SCRIPT $ARGS"
echo "----------------------------------------"

cd /users/rgupta25/fishy

python3 $SCRIPT $ARGS
EXIT_CODE=$?

echo "----------------------------------------"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
