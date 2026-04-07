# -*- coding: utf-8 -*-
"""
evaluate_model.py

Computes pixel-level classification metrics from Mask R-CNN test predictions:
  - Accuracy   : (TP + TN) / all pixels
  - Precision  : TP / (TP + FP)  -- when model says fish, how often correct?
  - Recall     : TP / (TP + FN)  -- of all fish pixels, how many found?
  - F1-score   : harmonic mean of precision and recall
  - ROC-AUC    : area under ROC curve using soft mask probability scores

All metrics are computed at the pixel level across the test set.
Each pixel is a binary classification: fish (1) vs background (0).

Why pixel-level?
  Your model is a segmentation model, not a classifier. The natural unit
  of evaluation is therefore the pixel. A predicted mask that covers the
  fish body but also bleeds into background has high recall but lower
  precision. A mask that is too conservative has high precision but lower
  recall. The F1 and ROC-AUC capture the balance between these.

ROC-AUC uses the soft (probabilistic) mask before thresholding, giving
thousands of score points per image even with only 11 test images. This
produces a meaningful curve without requiring more data.

Outputs:
  outputs/evaluation/
    metrics_summary.json      <- all five metrics, per-image breakdown
    metrics_summary.csv       <- same in CSV for spreadsheet use
    roc_curve.png             <- ROC curve plot
    precision_recall_curve.png
    per_image_metrics.png     <- bar chart of F1 per species

Usage:
  python scripts/python/evaluate_model.py

  # Use a different IoU threshold for TP/FP determination
  python scripts/python/evaluate_model.py --iou-threshold 0.75

  # Use a specific checkpoint
  python scripts/python/evaluate_model.py --checkpoint outputs/checkpoints/best_model.pth
"""

import json
import csv
import sys
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
ANN_DIR      = DATA_DIR / "annotations"
STD_DIR      = DATA_DIR / "standardized_images"
OUT_DIR      = PROJECT_ROOT / "outputs"
EVAL_DIR     = OUT_DIR / "evaluation"

ANN_JSON      = ANN_DIR / "annotations.json"
TEST_IDS_FILE = ANN_DIR / "test_ids.txt"
CHECKPOINT    = OUT_DIR / "checkpoints" / "best_model.pth"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Polygon -> mask utility
# ---------------------------------------------------------------------------

def polygon_to_mask(segmentation: list, H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


# ---------------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------------

def pixel_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict:
    """
    Compute pixel-level TP, FP, FN, TN and derived metrics.

    pred_bin : bool array [H, W]  -- predicted fish pixels
    gt_bin   : bool array [H, W]  -- ground truth fish pixels
    """
    TP = int(( pred_bin &  gt_bin).sum())
    FP = int(( pred_bin & ~gt_bin).sum())
    FN = int((~pred_bin &  gt_bin).sum())
    TN = int((~pred_bin & ~gt_bin).sum())
    total = TP + FP + FN + TN

    accuracy  = (TP + TN) / total          if total > 0  else 0.0
    precision = TP / (TP + FP)             if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN)             if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall /
                 (precision + recall))     if (precision + recall) > 0 else 0.0
    iou       = TP / (TP + FP + FN)        if (TP + FP + FN) > 0 else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "iou":       round(iou,       4),
    }


# ---------------------------------------------------------------------------
# ROC / PR curve utilities
# ---------------------------------------------------------------------------

def compute_roc(all_scores: np.ndarray, all_gt: np.ndarray) -> tuple:
    """
    Compute ROC curve points and AUC from flat pixel score/label arrays.

    Uses numpy-based approach (no sklearn required) for portability.
    """
    # Sort by descending score
    order      = np.argsort(-all_scores)
    scores_s   = all_scores[order]
    gt_s       = all_gt[order].astype(bool)

    n_pos = gt_s.sum()
    n_neg = len(gt_s) - n_pos

    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5

    # Accumulate TP and FP counts
    tp_cum = np.cumsum(gt_s)
    fp_cum = np.cumsum(~gt_s)

    tpr = tp_cum / n_pos
    fpr = fp_cum / n_neg

    # Prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # AUC via trapezoidal rule
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def compute_pr_curve(all_scores: np.ndarray, all_gt: np.ndarray) -> tuple:
    """Compute precision-recall curve and average precision."""
    order    = np.argsort(-all_scores)
    gt_s     = all_gt[order].astype(bool)
    n_pos    = gt_s.sum()

    if n_pos == 0:
        return np.array([0, 1]), np.array([1, 0]), 0.0

    tp_cum   = np.cumsum(gt_s)
    fp_cum   = np.cumsum(~gt_s)
    total_cum= np.arange(1, len(gt_s) + 1)

    prec = tp_cum / total_cum
    rec  = tp_cum / n_pos

    # Prepend (recall=0, precision=1)
    prec = np.concatenate([[1.0], prec])
    rec  = np.concatenate([[0.0], rec])

    ap = float(np.trapz(prec, rec))
    return rec, prec, ap


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(checkpoint_path: Path, iou_threshold: float) -> None:

    logger.info("=" * 70)
    logger.info("MASK R-CNN EVALUATION  --  pixel-level metrics")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Import from training script
    sys.path.insert(0, str(SCRIPT_DIR / "python"))
    try:
        from train_mask_rcnn import build_model, NUM_CLASSES, MASK_THRESHOLD
    except ImportError:
        logger.error("Cannot import train_mask_rcnn. Run from project root.")
        sys.exit(1)

    model = build_model(NUM_CLASSES).to(device)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Loaded: {checkpoint_path}  (epoch={ckpt['epoch']})")

    # ------------------------------------------------------------------
    # Load annotations and test IDs
    # ------------------------------------------------------------------
    with open(ANN_JSON) as f:
        coco = json.load(f)

    test_ids = [int(x) for x in TEST_IDS_FILE.read_text().split()]
    id_to_image = {img["id"]: img for img in coco["images"]}
    id_to_anns  = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        id_to_anns[ann["image_id"]].append(ann)

    logger.info(f"Test images: {len(test_ids)}")

    # ------------------------------------------------------------------
    # Run inference and collect per-image metrics
    # ------------------------------------------------------------------
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    per_image_results = []

    # For global ROC/PR curves: collect ALL pixel scores and labels
    all_soft_scores = []   # soft probability per pixel
    all_gt_labels   = []   # ground truth binary per pixel

    for img_id in test_ids:
        img_info = id_to_image.get(img_id)
        if img_info is None:
            logger.warning(f"Image ID {img_id} not in annotations.json")
            continue

        species  = Path(img_info["file_name"]).stem
        img_path = STD_DIR / img_info["file_name"]
        anns     = id_to_anns.get(img_id, [])

        logger.info(f"Processing: {species}")

        # Load image
        pil_img = Image.open(img_path).convert("RGB")
        H, W    = pil_img.size[1], pil_img.size[0]
        tensor  = TF.to_tensor(pil_img).unsqueeze(0).to(device)

        # Ground truth mask
        if anns:
            gt_mask = polygon_to_mask(anns[0]["segmentation"], H, W).astype(bool)
            has_gt  = True
        else:
            gt_mask = np.zeros((H, W), dtype=bool)
            has_gt  = False
            logger.warning(f"  No ground truth annotation for {species}")

        # Model prediction
        with torch.inference_mode():
            pred = model(tensor)[0]

        scores     = pred["scores"].cpu().numpy()
        soft_masks = pred["masks"].squeeze(1).cpu().numpy()  # [N, H, W] in [0,1]

        if len(scores) == 0:
            # No detection
            soft_mask = np.zeros((H, W), dtype=np.float32)
            pred_bin  = np.zeros((H, W), dtype=bool)
            detected  = False
        else:
            best_idx  = scores.argmax()
            soft_mask = soft_masks[best_idx]          # [H, W] probabilities
            pred_bin  = (soft_mask > MASK_THRESHOLD)  # binary
            detected  = True

        # Compute pixel metrics (skip if no ground truth)
        if has_gt:
            m = pixel_metrics(pred_bin, gt_mask)
            m["species"] = species
            m["detected"] = detected
            m["score"]    = round(float(scores.max()), 4) if detected else 0.0
            per_image_results.append(m)

            logger.info(
                f"  precision={m['precision']:.3f}  "
                f"recall={m['recall']:.3f}  "
                f"f1={m['f1']:.3f}  "
                f"iou={m['iou']:.3f}  "
                f"acc={m['accuracy']:.3f}"
            )

            # Collect pixels for global ROC
            # Sample uniformly to avoid memory issues (max 200k pixels per image)
            flat_soft = soft_mask.flatten()
            flat_gt   = gt_mask.flatten().astype(np.float32)
            if len(flat_soft) > 200_000:
                idx = np.random.RandomState(42).choice(
                    len(flat_soft), 200_000, replace=False
                )
                flat_soft = flat_soft[idx]
                flat_gt   = flat_gt[idx]
            all_soft_scores.append(flat_soft)
            all_gt_labels.append(flat_gt)
        else:
            logger.info(f"  Skipped (no ground truth)")

    if not per_image_results:
        logger.error("No results to evaluate.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    keys = ["accuracy", "precision", "recall", "f1", "iou"]
    macro = {k: round(float(np.mean([r[k] for r in per_image_results])), 4) for k in keys}

    # Micro (pooled TP/FP/FN/TN across all images)
    sum_TP = sum(r["TP"] for r in per_image_results)
    sum_FP = sum(r["FP"] for r in per_image_results)
    sum_FN = sum(r["FN"] for r in per_image_results)
    sum_TN = sum(r["TN"] for r in per_image_results)
    total  = sum_TP + sum_FP + sum_FN + sum_TN

    micro = {
        "accuracy":  round((sum_TP + sum_TN) / total, 4) if total > 0 else 0.0,
        "precision": round(sum_TP / (sum_TP + sum_FP), 4) if (sum_TP+sum_FP) > 0 else 0.0,
        "recall":    round(sum_TP / (sum_TP + sum_FN), 4) if (sum_TP+sum_FN) > 0 else 0.0,
    }
    micro["f1"] = round(
        2 * micro["precision"] * micro["recall"] /
        (micro["precision"] + micro["recall"]), 4
    ) if (micro["precision"] + micro["recall"]) > 0 else 0.0
    micro["iou"] = round(
        sum_TP / (sum_TP + sum_FP + sum_FN), 4
    ) if (sum_TP + sum_FP + sum_FN) > 0 else 0.0

    # Global ROC-AUC
    all_scores_flat = np.concatenate(all_soft_scores)
    all_gt_flat     = np.concatenate(all_gt_labels)
    fpr, tpr, roc_auc = compute_roc(all_scores_flat, all_gt_flat)
    rec_c, prec_c, avg_prec = compute_pr_curve(all_scores_flat, all_gt_flat)

    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info("Macro-averaged (mean of per-image scores):")
    for k in keys:
        logger.info(f"  {k:<12}: {macro[k]:.4f}")
    logger.info("")
    logger.info("Micro-averaged (pooled pixels across all images):")
    for k in ["accuracy", "precision", "recall", "f1", "iou"]:
        logger.info(f"  {k:<12}: {micro[k]:.4f}")
    logger.info("")
    logger.info(f"ROC-AUC         : {roc_auc:.4f}")
    logger.info(f"Avg Precision   : {avg_prec:.4f}")

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------
    summary = {
        "checkpoint":    str(checkpoint_path),
        "n_test_images": len(per_image_results),
        "iou_threshold": iou_threshold,
        "macro_averaged": macro,
        "micro_averaged": micro,
        "roc_auc":       round(roc_auc, 4),
        "average_precision": round(avg_prec, 4),
        "per_image": per_image_results,
    }
    json_path = EVAL_DIR / "metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved: {json_path}")

    # ------------------------------------------------------------------
    # Save CSV summary
    # ------------------------------------------------------------------
    csv_path = EVAL_DIR / "metrics_summary.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["species", "accuracy", "precision", "recall",
                      "f1", "iou", "score", "detected",
                      "TP", "FP", "FN", "TN"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_image_results)
        # Summary rows
        writer.writerow({})
        writer.writerow({"species": "MACRO_AVG", **macro})
        writer.writerow({"species": "MICRO_AVG", **micro})
        writer.writerow({"species": "ROC_AUC",
                         "accuracy": roc_auc, "precision": avg_prec})
    logger.info(f"Saved: {csv_path}")

    # ------------------------------------------------------------------
    # Plot 1: ROC curve
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.plot(fpr, tpr, color="#4fc3f7", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#4a4d5e", lw=1, ls="--",
            label="Random classifier (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", color="#e8e8f0")
    ax.set_ylabel("True Positive Rate", color="#e8e8f0")
    ax.set_title("ROC Curve -- Pixel-level Fish Segmentation",
                 color="#e8e8f0", fontsize=12)
    ax.legend(frameon=False, labelcolor="#e8e8f0")
    ax.tick_params(colors="#4a4d5e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4a4d5e")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.fill_between(fpr, tpr, alpha=0.08, color="#4fc3f7")
    roc_path = EVAL_DIR / "roc_curve.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info(f"Saved: {roc_path}")

    # ------------------------------------------------------------------
    # Plot 2: Precision-Recall curve
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.plot(rec_c, prec_c, color="#aed581", lw=2,
            label=f"PR curve (AP = {avg_prec:.3f})")
    ax.set_xlabel("Recall", color="#e8e8f0")
    ax.set_ylabel("Precision", color="#e8e8f0")
    ax.set_title("Precision-Recall Curve -- Pixel-level Fish Segmentation",
                 color="#e8e8f0", fontsize=12)
    ax.legend(frameon=False, labelcolor="#e8e8f0")
    ax.tick_params(colors="#4a4d5e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4a4d5e")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.fill_between(rec_c, prec_c, alpha=0.08, color="#aed581")
    pr_path = EVAL_DIR / "precision_recall_curve.png"
    fig.savefig(pr_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info(f"Saved: {pr_path}")

    # ------------------------------------------------------------------
    # Plot 3: Per-species F1 bar chart
    # ------------------------------------------------------------------
    species_names = [r["species"] for r in per_image_results]
    f1_scores     = [r["f1"]      for r in per_image_results]
    prec_scores   = [r["precision"] for r in per_image_results]
    rec_scores    = [r["recall"]    for r in per_image_results]

    x     = np.arange(len(species_names))
    width = 0.28

    fig, ax = plt.subplots(figsize=(max(12, len(species_names) * 1.2), 6),
                           facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.bar(x - width, prec_scores, width, label="Precision", color="#4fc3f7", alpha=0.85)
    ax.bar(x,         f1_scores,   width, label="F1",        color="#aed581", alpha=0.85)
    ax.bar(x + width, rec_scores,  width, label="Recall",    color="#ffb74d", alpha=0.85)

    ax.set_xticks(x)
    short_names = [s.replace("Acanthurus_", "A.").replace("Acanthurus ", "A.")
                    .replace("Ctenochaetus_", "Ct.").replace("Ctenochaetus ", "Ct.")
                    .replace("Naso_", "N.").replace("Naso ", "N.")
                    .replace("Prionurus_", "Pr.").replace("Prionurus ", "Pr.")
                    .replace("Zebrasoma_", "Z.").replace("Zebrasoma ", "Z.")
                   for s in species_names]
    ax.set_xticklabels(short_names, rotation=45, ha="right",
                       color="#e8e8f0", fontsize=8)
    ax.set_ylabel("Score", color="#e8e8f0")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-species Precision / F1 / Recall (Test Set)",
                 color="#e8e8f0", fontsize=12)
    ax.legend(frameon=False, labelcolor="#e8e8f0")
    ax.tick_params(colors="#4a4d5e")
    ax.axhline(macro["f1"], color="#e8e8f0", lw=0.8, ls="--", alpha=0.5)
    ax.text(len(species_names) - 0.5, macro["f1"] + 0.01,
            f"macro F1={macro['f1']:.2f}", color="#e8e8f0", fontsize=8, ha="right")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4a4d5e")

    fig.tight_layout()
    bar_path = EVAL_DIR / "per_image_metrics.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    logger.info(f"Saved: {bar_path}")

    # ------------------------------------------------------------------
    # Final summary print
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Images evaluated : {len(per_image_results)}")
    logger.info(f"  Accuracy         : {micro['accuracy']:.4f}  "
                f"({micro['accuracy']*100:.1f}% of pixels correctly classified)")
    logger.info(f"  Precision        : {micro['precision']:.4f}  "
                f"(when model predicts fish, correct {micro['precision']*100:.1f}% of time)")
    logger.info(f"  Recall           : {micro['recall']:.4f}  "
                f"({micro['recall']*100:.1f}% of actual fish pixels found)")
    logger.info(f"  F1-score         : {micro['f1']:.4f}")
    logger.info(f"  IoU              : {micro['iou']:.4f}")
    logger.info(f"  ROC-AUC          : {roc_auc:.4f}  "
                f"({'excellent' if roc_auc > 0.95 else 'good' if roc_auc > 0.85 else 'moderate'})")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  {EVAL_DIR}/metrics_summary.json")
    logger.info(f"  {EVAL_DIR}/metrics_summary.csv")
    logger.info(f"  {EVAL_DIR}/roc_curve.png")
    logger.info(f"  {EVAL_DIR}/precision_recall_curve.png")
    logger.info(f"  {EVAL_DIR}/per_image_metrics.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Mask R-CNN segmentation with pixel-level metrics",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINT,
        help="Path to model checkpoint (default: outputs/checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5,
        help="IoU threshold for TP/FP in detection metrics (default: 0.5)",
    )
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.iou_threshold)


if __name__ == "__main__":
    main()
