# CNN-Based Morphological Analysis and Phylogenetic Inference in Surgeonfish

> A graduate research project in the **Dornburg Lab** at UNC Charlotte training a YOLOv8 convolutional neural network to automatically detect and classify six species of surgeonfish from underwater imagery, with the broader goal of investigating whether CNN-learned morphological features carry phylogenetic signal.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)]()
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00D9FF)]()
[![License](https://img.shields.io/badge/License-GPL--3.0-blue)]()

---

## Table of Contents
- [Motivation](#motivation)
- [Approach](#approach)
- [Repository Structure](#repository-structure)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)

---

## Motivation

Manual identification of fish species from images is slow, subjective, and difficult to scale. Automating the task with a convolutional neural network frees researchers to analyze larger image sets and — more interestingly — lets us quantify morphological features that are hard to measure by hand. If those learned features correlate with known phylogenetic relationships, that's a novel quantitative tool for morphometric evolutionary biology.

This project tackles two questions:

1. **Can a CNN reliably detect and classify six species of surgeonfish** from underwater images with limited training data?
2. **Do the CNN's learned feature representations recover known phylogenetic relationships** among those species?

---

## Approach

A YOLOv8 detector is fine-tuned on a preprocessed, curated dataset of 83 underwater images across six surgeonfish species. Training runs on UNCC's HPC cluster using GPU partitions via SLURM. After training, per-image feature embeddings are extracted from the trained backbone and compared to the known surgeonfish phylogeny.

---

## Repository Structure

```
Surgeonfish_Neural_Network_Phylogenetics/
├── data/                      # Raw and preprocessed images + annotations
├── scripts/                   # Preprocessing, training, evaluation scripts
│   ├── preprocess.py          # Illumination/orientation/background standardization
│   ├── train.py               # YOLOv8 training launch
│   ├── evaluate.py            # Precision/recall/mAP + confusion matrix
│   └── slurm/                 # SLURM batch scripts for HPC training
├── outputs/                   # Model checkpoints, training curves, predictions
├── reports/                   # Analysis write-ups and figures
├── README.md
└── LICENSE
```

---

## Pipeline

```
┌─────────────────┐
│  Raw Images     │  83 underwater images, 6 species
│   (+ labels)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  1. Preprocessing                        │
│     • Illumination correction             │
│     • Orientation normalization           │
│     • Background noise reduction          │
│     • Data augmentation (flip, jitter)    │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  2. Dataset Prep                         │
│     • YOLOv8 annotation format            │
│     • Stratified train/val/test split     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  3. Training (SLURM + GPU partitions)    │
│     • YOLOv8m fine-tuned from COCO         │
│     • AdamW + cosine LR                    │
│     • Hyperparameter sweep                 │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  4. Evaluation                           │
│     • Precision / Recall / mAP@0.5        │
│     • Per-class confusion matrix          │
│     • Error analysis                      │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  5. Phylogenetic Comparison (ongoing)    │
│     • CNN feature embeddings              │
│     • Compare clustering to known tree    │
└─────────────────────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/guptrishi01/Surgeonfish_Neural_Network_Phylogenetics.git
cd Surgeonfish_Neural_Network_Phylogenetics

# Create conda environment
conda create -n surgeonfish python=3.10 -y
conda activate surgeonfish

# Install dependencies
pip install ultralytics torch torchvision opencv-python pandas matplotlib scikit-learn
```

---

## Usage

### 1. Preprocess images
```bash
python scripts/preprocess.py --input data/raw --output data/processed
```

### 2. Train locally (small test run)
```bash
python scripts/train.py --data surgeonfish.yaml --epochs 50 --batch 8
```

### 3. Train on HPC (recommended)
```bash
sbatch scripts/slurm/train_gpu.sh
```

### 4. Evaluate
```bash
python scripts/evaluate.py --weights outputs/best.pt --data surgeonfish.yaml
```

---

## Results

Training and evaluation in progress. Interim metrics will be added here as hyperparameter sweeps complete.

Planned metrics:
- Per-class precision / recall / F1
- mAP@0.5 and mAP@0.5:0.95
- Confusion matrix across all six species
- CNN embedding clustering compared to phylogenetic distance matrix

---

## Roadmap

- [x] Define species set and collect images
- [x] Build preprocessing pipeline
- [x] Annotate dataset in YOLOv8 format
- [x] Set up HPC training on SLURM
- [ ] Complete hyperparameter sweep
- [ ] Extract feature embeddings from trained backbone
- [ ] Compare embedding clusters to known phylogeny
- [ ] Write-up for publication / conference

---

## Acknowledgements

I would like to thank **Dr. Alex Dornburg** and the Dornburg Lab at UNC Charlotte for bringing me onto this project. This work leverages UNC Charlotte's High-Performance Computing cluster for all training and evaluation.

---

## License

GPL-3.0 — see [LICENSE](LICENSE).
