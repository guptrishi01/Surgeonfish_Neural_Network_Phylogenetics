#!/bin/bash
#SBATCH --job-name=verify_env
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err

# Load modules matching the actual GPU driver
module load cuda/12.8
module load gcc/13.2.0

# Source conda properly (not 'conda activate' which needs init)
source /users/rgupta25/.conda/etc/profile.d/conda.sh
conda activate surgeonfish

python -c "
import torch, torchvision, cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
model = maskrcnn_resnet50_fpn(weights='DEFAULT')
print('Mask R-CNN: OK')
import sam2
print('SAM 2: OK')
print('ALL CHECKS PASSED')
"
