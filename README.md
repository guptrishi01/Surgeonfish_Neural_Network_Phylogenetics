# Surgeonfish Visual Phenomics and Phylogenetic Inference
 
> A computational biology research project developing a fully automated computer vision pipeline to extract visual phenotypic features from lateral photographs of surgeonfish (family Acanthuridae) and test whether those features carry phylogenetic signal — i.e., whether morphologically similar species are also more closely related.
 
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C?logo=pytorch&logoColor=white)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)]()
[![License](https://img.shields.io/badge/License-GPL--3.0-blue)]()
 
---

## Table of Contents
 
- [Scientific Motivation](#scientific-motivation)
- [Approach](#approach)
- [Repository Structure](#repository-structure)
- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Key Results](#key-results)
- [Acknowledgements](#acknowledgements)
---

## Scientific Motivation
 
Surgeonfish (family Acanthuridae) display remarkable diversity in colour pattern, from the vivid blue-and-yellow of *Paracanthurus hepatus* to the cryptic brown stripes of *Ctenochaetus striatus*. A longstanding question in evolutionary biology is whether these visual traits evolve primarily under phylogenetic constraint — meaning closely related species tend to look alike — or primarily under ecological selection, which would produce convergent appearances across distantly related lineages.
 
This project addresses that question quantitatively across 63 species spanning all six extant Acanthuridae genera: *Acanthurus* (31 species), *Ctenochaetus* (9), *Naso* (14), *Paracanthurus* (1), *Prionurus* (3), and *Zebrasoma* (6). The known molecular phylogeny from the Fish Tree of Life (fishtreeoflife.org), a time-calibrated Newick tree built from nine molecular loci, serves as the reference topology against which visual similarity is compared.
 
A key biological complication motivating this work is that *Acanthurus* is paraphyletic with respect to *Ctenochaetus*: several *Ctenochaetus* species are nested inside the *Acanthurus* clade in the molecular tree. The pipeline tests whether this paraphyly is also reflected in visual similarity scores, providing a direct check on whether computer vision can recover known molecular evolutionary relationships.
 
---

## Approach
 
The pipeline extracts a 99-dimensional feature vector from each species' standardised lateral photograph using classical computer vision methods — colour histograms, dominant colour clustering, dorsal-ventral colour gradients, Gabor texture filters, and Local Binary Pattern histograms — all computed exclusively from within a predicted fish body mask to exclude background. Pairwise visual distance matrices are then compared to the molecular phylogenetic distance matrix using a Mantel test to quantify phylogenetic signal.
 
A Mask R-CNN model (ResNet-50 FPN backbone, fine-tuned from COCO) is trained on 63 manually curated and SAM-2-annotated lateral images to produce pixel-tight body masks for all 63 species. Feature extraction, distance matrix construction, and statistical comparison are fully automated.
 
---

## Repository Structure

```
Surgeonfish_Neural_Network_Phylogenetics/
├── data/
│   ├── raw_images/              # Manually screened source images, one per species
│   │   ├── Acanthurus/
│   │   ├── Ctenochaetus/
│   │   ├── Naso/
│   │   ├── Paracanthurus/
│   │   ├── Prionurus/
│   │   └── Zebrasoma/
│   ├── standardized_images/     # 1024x1024 PNG, mid-grey padding, lossless
│   │   ├── Acanthurus/
│   │   ├── Ctenochaetus/
│   │   ├── Naso/
│   │   ├── Paracanthurus/
│   │   ├── Prionurus/
│   │   └── Zebrasoma/
│   ├── annotations/
│   │   ├── split_summary.json   # COCO-format, 63 annotated images
│   │   ├── annotations.json     # COCO-format, train & test split image IDs
│   │   ├── train_ids.txt        # 43 image IDs
│   │   ├── val_ids.txt          # 10 image IDs
│   │   └── test_ids.txt         # 11 image IDs
│   ├── features/
│   │   ├── features.csv         # 63 species x 99 features
│   │   ├── feature_names.txt    # Feature names
│   │   ├── extraction_log.csv   # 63 species x 5 features
│   │   └── features.json        # Same with metadata
│   └── phylogeny/
│       |── Acanthuridae_phylogram.tre  # Acanthuridae phylogenetic tree
│       └── Acanthuridae_timetree.tre   # Time-calibrated Newick (Fish Tree of Life)
│
├── outputs/
│   ├── checkpoints/
│   │   └── best_model.pth   # Best Mask R-CNN checkpoint (val mask AP)
│   ├── all_predictions/     # Binary mask PNGs for all 64 images
│   ├── predictions/         # Binary mask and overlay for Naso annulatus
│   ├── test_predictions/    # Binary mask PNGs for all 22 images
│   ├── val_predictions/     # Binary mask PNGs for 20 images
│   ├── evaluation/
│   │   ├── per_image_metrics.png
│   │   ├── precision_recall_curve.png
│   │   ├── roc_curve.png
│   │   ├── metrics_summary.json
│   │   └── metrics_summary.csv
│   ├── validation/
│   │   └── feature_validation_report.csv
│   ├── visualisation/
│   │   ├── report.html          # Visual feature inspection report
│   │   └── figures/             # 63 x 8-panel PNG inspection figures
│   ├── distance_matrices/
│   │   ├── visual_distance_matrix_aligned.csv
│   │   ├── visual_distance_matrix_featureselected_aligned.csv
│   │   ├── patristic_distance_matrix_aligned.csv
│   │   ├── visual_distance_heatmap.png
│   │   ├── patristic_distance_heatmap.png
│   │   ├── pca_variance_explained.png
│   │   ├── tree_tip_matching_audit.csv
│   │   └── distance_matrix_summary.json
│   └── phylogenetic_analysis/
│       ├── mantel_test_results.json
│       ├── mantel_permutation_distribution_pca.png
│       ├── mantel_permutation_distribution_featureselected.png
│       ├── visual_dendrogram.png
│       ├── tanglegram.png
│       ├── feature_mantel_correlations.csv
│       ├── feature_mantel_correlations.png
│       └── analysis_summary.json
│
├── scripts/
│   ├── python/
│   │   ├── compare_to_phylogeny.py
│   │   ├── build_distance_matrix.py
│   │   ├── visualize_features.py
│   │   ├── validate_features.py
│   │   ├── extract_features.py
│   │   ├── train_mask_rcnn.py
│   │   ├── evaluate_model.py
│   │   ├── fix_annotation.py
│   │   ├── prepare_splits.py
│   │   ├── generate_annotations.py
│   │   └── standardize_images.py
│   └── bash/
│       ├── run_scripts.sh
│       ├── create_phylogenetic_distances.sh
│       ├── visualize_features.sh
│       ├── validate_features.sh
│       ├── extract_features.sh
│       ├── evaluate_model.sh
│       ├── train_mask_rcnn.sh
│       ├── verify_env.sh
│       ├── setup_env.bash
│       ├── generate_annotations.sh
│       └── standardize_images.sh
├── reports/
│   ├── annotation_log.csv
│   └── standardization_log.csv
```

---

## Pipeline
The pipeline runs in nine sequential steps, each as a standalone Python script.


```
┌─────────────────┐
│  Raw Images     │  83 underwater images, 6 species
│   (+ labels)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  1. standardize_images.py               │
│     • Resize to 1024x1024               │
│     • EXIF rotation                     │
│     • Mid-grey padding                  │
│     • Lossless PNG                      │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  2. generate_annoations.py              │
│     • SAM-2 automated segmentation      │
│     • COCO-format annotations.json      │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  3. prepare_splits.py                   │
│     • Stratified genus-balanced split   │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  4. train_mask_rcnn.py                  │
│     • Phase 1: Frozen backbone          │
│     • Phase 2: Unfrozen backbone        │
│     • Mask AP @ IoU 0.5:0.95 = 0.66     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  5. evaluate_model.py                   │
│     • Pixel-level binary metrics        │
│     • ROC-AUC, per-image mask quality   │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  6. extract_features.py                 │
│     • 99-dimensional feature vector     │
│     • Hue/Saturation/Value histogram    │
│     • Dorsal/ventral colour gradient    │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  7. visualize_features.py               │
│     • 8-panel inspection figure         │
│     • Panels: Mask overlay, S/V scatter │
│     • Gabor heatmap, LBP map, ...       │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  8. build_distance_matrix.py            │
│     • PCA visual distance matrix        │
│     • Feature-selected distance matrix  │
│     • Patristic distance matrix         │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  9. compare_to_phylogeny.py             │
│     • Mantel test: matrix vs matrix     │
│     • Visual similarity dendogram       │
│     • Tanglegram and Mantel correlations│
└─────────────────────────────────────────┘
```

---

## Installation
This project runs on the `gal-i9` HPC cluster (NVIDIA GTX 1080 Ti, CUDA 12.8) via SLURM. 
All GPU jobs are submitted through `slurm/run_script.sh`.

```bash
# Clone the repository
git clone https://github.com/guptrishi01/Surgeonfish_Neural_Network_Phylogenetics.git
cd Surgeonfish_Neural_Network_Phylogenetics

# Create conda environment
conda env create -f environment.yml

conda create -n surgeonfish python=3.10 -y
conda activate surgeonfish


# Install dependencies
pip install torch==2.4.0+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python scikit-image scikit-learn scipy matplotlib pandas
pip install sam2 dendropy
 
# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Usage
Each script can be run directly or submitted to SLURM via the universal batch script. All paths default to the repository layout and require no arguments when run from the project root.

```bash
# Step 1: Standardize images
python scripts/python/standardize_images.py
 
# Step 2-3: Annotate and prepare splits
python scripts/python/generate_annotations.py
python scripts/python/prepare_splits.py
 
# Step 4-5: Train and evaluate Mask R-CNN (submit to GPU node)
sbatch slurm/run_script.sh scripts/python/train_mask_rcnn.py
sbatch slurm/run_script.sh scripts/python/evaluate_model.py
 
# Step 6-7: Extract features and inspect visually
python scripts/python/extract_features.py
python scripts/python/visualise_features.py
 
# Step 8-9: Build distance matrices and run phylogenetic comparison
python scripts/python/build_distance_matrix.py
python scripts/python/compare_to_phylogeny.py --n-permutations 99999
 
# Inspect the tree tip matching audit before running step 8:
python scripts/python/build_distance_matrix.py --inspect-tree
```

---

## Results

The pipeline successfully produced pixel-tight body masks, a validated 99-feature visual phenotype matrix, and a complete phylogenetic comparison for 48 species matched to the Fish Tree of Life.
 
**Mask R-CNN performance.** Training in two phases (frozen then unfrozen backbone) on 43 images achieved test mask AP of 0.66 at IoU 0.5:0.95. Pixel-level binary classification reached ROC-AUC > 0.95 across the test set. Masks were visually confirmed for all 63 species via the inspection report.
 
**Phylogenetic signal — PCA feature space.** A Mantel test comparing the full 31-component PCA visual distance matrix against the molecular patristic distance matrix yielded r = 0.055, p = 0.221 (99,999 permutations, n = 48 species). This result is non-significant and falls below the minimum detectable effect size (r = 0.074, α = 0.05, power = 0.80). PCA aggregates all 99 features including 40 with negative correlation to phylogenetic distance, which dilutes the colour signal.
 
**Phylogenetic signal — feature-selected space.** A second Mantel test restricted to the 37 features from the four groups with positive mean phylogenetic signal (hue histogram, saturation histogram, dorsal-ventral gradient, entropy) yielded r = 0.137, p = 0.027 (Spearman ρ = 0.134, p = 0.026). This result is significant at α = 0.05, confirmed by both Pearson and Spearman statistics across 99,999 permutations.
 
**Most phylogenetically informative features.** Individual feature Mantel correlations identified the blue-green through cyan hue range (OpenCV hue bins 70–100, r = 0.172–0.264) and the dorsal hue mean (r = 0.211) as the strongest carriers of phylogenetic signal. Pattern entropy (r = 0.146) and countershading magnitude (dv_diff_val_mean, r = 0.142) were also significant. Texture features (LBP, Gabor) and overall brightness (val_hist) carried no phylogenetic signal and actively cancelled the colour signal in the composite PCA.
 
**Biological interpretation.** The significant signal concentrated in the blue-green hue range reflects the systematic differentiation between the grey-blue *Naso* clade, the darker *Acanthurus*/*Ctenochaetus* complex, and the more variable *Zebrasoma* species. The *Acanthurus*–*Ctenochaetus* paraphyly identified in the molecular tree is confirmed in the patristic distance matrix: the minimum *Acanthurus*–*Ctenochaetus* patristic distance (16.5 Myr) is shorter than the mean within-*Acanthurus* distance (28.4 Myr), and this nested structure is partially reflected in the visual dendrogram.
 
| Metric | Value |
|---|---|
| Species in visual matrix | 63 |
| Species matched to phylogeny | 48 |
| Mask R-CNN test mask AP (IoU 0.5:0.95) | 0.66 |
| Mantel r (full PCA, 99,999 perms) | 0.055 (p = 0.221, NS) |
| Mantel r (feature-selected, 99,999 perms) | 0.137 (p = 0.027, *) |
| Strongest individual feature | hue_bin_090_100 (r = 0.264) |
| Strongest feature group | dorsal_ventral + entropy |

---

## Acknowledgements

This project was conducted in the **Dornburg Lab** at the University of North Carolina Charlotte. Phylogenetic reference data were obtained from the Fish Tree of Life (fishtreeoflife.org). All training and computation were performed on the UNC Charlotte HPC cluster.

---
