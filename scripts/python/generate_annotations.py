# -*- coding: utf-8 -*-
"""
generate_annotations.py

Automated annotation of standardized surgeonfish images using SAM 2.
Produces a COCO-format annotations.json ready for Mask R-CNN training.

Strategy:
  Primary prompt  -- single point at image center (512, 512).
      Your standardization pipeline centers and pads every image to
      1024x1024, so the center pixel almost always lands on the fish body.

  Fallback prompt -- if the center-point mask covers less than MIN_MASK_AREA
      of the image (meaning the point landed on padding or background),
      the script retries with a 3x3 grid of points in the central 50%
      of the image and picks the mask with the highest SAM score.

  Quality filter  -- masks below MIN_MASK_AREA or above MAX_MASK_AREA are
      flagged in the log as LOW_CONFIDENCE and excluded from the COCO JSON.
      You can inspect these manually and re-include them if they look correct.

Output layout:
  data/
    annotations/
      annotations.json          <- COCO-format, ready for Mask R-CNN
      masks/                    <- one binary PNG mask per image (for QC)
        Acanthurus/
          Acanthurus_mata.png
        ...
      annotation_log.csv        <- per-image: score, mask_area, strategy used

COCO JSON structure produced:
  {
    "info":        { ... },
    "licenses":    [],
    "categories":  [{"id": 1, "name": "fish", "supercategory": "animal"}],
    "images":      [{"id": int, "file_name": str, "width": int, "height": int}],
    "annotations": [{
        "id":           int,
        "image_id":     int,
        "category_id":  1,
        "segmentation": [[x1,y1,x2,y2,...]]  <- polygon in COCO RLE format
        "area":         float,
        "bbox":         [x, y, w, h],
        "iscrowd":      0
    }]
  }

Installation (add to your surgeonfish conda environment):
  pip install sam2
  # SAM 2 downloads model weights automatically from HuggingFace on first run.
  # Requires Python >= 3.10, torch >= 2.3.1

  # If you have a GPU:
  #   conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
  # CPU only (slower, ~10s per image):
  #   pip install torch torchvision

Usage:
  # Annotate all standardized images
  python scripts/python/generate_annotations.py

  # Dry run -- shows what would be processed without running SAM
  python scripts/python/generate_annotations.py --dry-run

  # Annotate a single image and save its mask for inspection
  python scripts/python/generate_annotations.py --single data/standardized_images/Acanthurus/Acanthurus_mata.png

  # Use a lighter/faster model (slightly lower quality)
  python scripts/python/generate_annotations.py --model sam2-hiera-small
"""

import os
import sys
import csv
import json
import logging
import argparse
import torchvision
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
STD_DIR      = DATA_DIR / "standardized_images"
ANN_DIR      = DATA_DIR / "annotations"
MASK_DIR     = ANN_DIR / "masks"
REPORTS_DIR  = PROJECT_ROOT / "reports"

SUPPORTED_EXT = {".png", ".jpg", ".jpeg"}

GENERA = ["Acanthurus", "Ctenochaetus", "Naso", "Paracanthurus", "Prionurus", "Zebrasoma"]

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------

# Minimum fraction of image area the fish mask must cover.
# Fish should occupy at least 10% of a 1024x1024 image after standardization.
MIN_MASK_AREA = 0.10

# Maximum fraction -- if the mask covers >90% it probably leaked into background.
MAX_MASK_AREA = 0.90

# Minimum SAM confidence score to trust the mask.
MIN_SAM_SCORE = 0.70

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
# SAM 2 loader
# ---------------------------------------------------------------------------

def load_sam2(model_id: str = "facebook/sam2-hiera-large", device: str = "auto"):
    """
    Load SAM 2 predictor from HuggingFace.

    model_id options (larger = better quality, slower):
      facebook/sam2-hiera-large   -- best quality  (~2.4 GB)
      facebook/sam2-hiera-base-plus               (~0.9 GB)
      facebook/sam2-hiera-small                   (~0.5 GB)
      facebook/sam2-hiera-tiny    -- fastest       (~0.4 GB)

    Weights are cached locally after first download (~/.cache/huggingface/).
    """
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        logger.error(
            "SAM 2 is not installed. Run: pip install sam2\n"
            "Also install PyTorch >= 2.3.1: https://pytorch.org/get-started/locally/"
        )
        sys.exit(1)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Using GPU (CUDA)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Silicon GPU (MPS)")
        else:
            device = "cpu"
            logger.info("Using CPU -- expect ~10s per image")

    logger.info(f"Loading SAM 2 model: {model_id}")
    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    logger.info("SAM 2 loaded")
    return predictor, device


# ---------------------------------------------------------------------------
# Core annotation function for a single image
# ---------------------------------------------------------------------------

def annotate_image(
    predictor,
    image_path: Path,
    device: str,
    min_mask_area: float = MIN_MASK_AREA,
    max_mask_area: float = MAX_MASK_AREA,
    min_score: float = MIN_SAM_SCORE,
) -> dict:
    """
    Run SAM 2 on one standardized image and return annotation data.

    Returns a dict with:
      status       : "OK" | "LOW_CONFIDENCE" | "FAILED"
      mask         : numpy bool array [H, W]  (None if failed)
      score        : SAM confidence score
      mask_area_frac: fraction of image covered by mask
      strategy     : "center_point" | "grid_fallback"
      bbox         : [x, y, w, h]
      message      : human-readable note
    """

    result = {
        "status": "FAILED",
        "mask": None,
        "score": 0.0,
        "mask_area_frac": 0.0,
        "strategy": "",
        "bbox": [],
        "message": "",
    }

    # Load image
    image_np = np.array(Image.open(image_path).convert("RGB"))
    H, W = image_np.shape[:2]
    img_area = H * W

    # Context manager handles torch.inference_mode and autocast appropriately
    ctx_device = device if device != "cpu" else "cpu"

    try:
        if ctx_device == "cpu":
            ctx = torch.inference_mode()
            use_autocast = False
        else:
            ctx = torch.inference_mode()
            use_autocast = True

        with torch.inference_mode():
            if use_autocast:
                with torch.autocast(ctx_device, dtype=torch.bfloat16):
                    predictor.set_image(image_np)
                    best_mask, best_score, strategy = _run_prompts(
                        predictor, W, H, img_area, min_mask_area
                    )
            else:
                predictor.set_image(image_np)
                best_mask, best_score, strategy = _run_prompts(
                    predictor, W, H, img_area, min_mask_area
                )

    except Exception as e:
        result["message"] = f"SAM inference error: {e}"
        return result

    if best_mask is None:
        result["message"] = "No valid mask found after all prompt strategies"
        return result

    mask_frac = best_mask.sum() / img_area
    result["mask"] = best_mask
    result["score"] = float(best_score)
    result["mask_area_frac"] = float(mask_frac)
    result["strategy"] = strategy

    # Compute bounding box from mask
    ys, xs = np.where(best_mask)
    if len(xs) == 0:
        result["message"] = "Empty mask"
        return result

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    result["bbox"] = [x_min, y_min, x_max - x_min, y_max - y_min]

    # Quality gate
    if best_score < min_score:
        result["status"] = "LOW_CONFIDENCE"
        result["message"] = (
            f"SAM score {best_score:.3f} below threshold {min_score}. "
            "Inspect mask manually."
        )
        return result

    if mask_frac < min_mask_area:
        result["status"] = "LOW_CONFIDENCE"
        result["message"] = (
            f"Mask covers only {mask_frac:.1%} of image "
            f"(minimum {min_mask_area:.1%}). Fish may not be centered."
        )
        return result

    if mask_frac > max_mask_area:
        result["status"] = "LOW_CONFIDENCE"
        result["message"] = (
            f"Mask covers {mask_frac:.1%} -- likely leaked into background."
        )
        return result

    result["status"] = "OK"
    result["message"] = f"score={best_score:.3f}, area={mask_frac:.1%}, strategy={strategy}"
    return result


def _run_prompts(predictor, W, H, img_area, min_mask_area):
    """
    Try prompt strategies in order, return (mask, score, strategy_name).

    Strategy 1 -- center point (512, 512 for 1024x1024 images).
    Strategy 2 -- 3x3 grid of points in the central 50% of the image,
                  pick the highest-scoring mask.
    """
    cx, cy = W // 2, H // 2

    # -- Strategy 1: single center point --
    mask, score = _predict_single_point(predictor, cx, cy)
    if mask is not None and mask.sum() / img_area >= min_mask_area:
        return mask, score, "center_point"

    # -- Strategy 2: 3x3 grid fallback --
    grid_masks = []
    for gx in [W * 3 // 8, W // 2, W * 5 // 8]:
        for gy in [H * 3 // 8, H // 2, H * 5 // 8]:
            m, s = _predict_single_point(predictor, gx, gy)
            if m is not None:
                grid_masks.append((m, s))

    if not grid_masks:
        return None, 0.0, "grid_fallback"

    # Pick the largest valid mask (most likely to be the fish body)
    grid_masks.sort(key=lambda t: t[0].sum(), reverse=True)
    best_mask, best_score = grid_masks[0]
    return best_mask, best_score, "grid_fallback"


def _predict_single_point(predictor, px, py):
    """Run SAM 2 with a single foreground point, return (mask, score)."""
    try:
        point_coords = np.array([[px, py]])
        point_labels = np.array([1])   # 1 = foreground

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,     # SAM returns 3 candidate masks
        )
        # Pick the highest-scoring candidate
        best_idx  = scores.argmax()
        best_mask = masks[best_idx].astype(bool)
        best_score = float(scores[best_idx])
        return best_mask, best_score
    except Exception:
        return None, 0.0


# ---------------------------------------------------------------------------
# Mask -> COCO polygon conversion
# ---------------------------------------------------------------------------

def mask_to_coco_polygon(mask: np.ndarray) -> list:
    """
    Convert a binary mask to a COCO-format polygon list.

    COCO segmentation format: [[x1, y1, x2, y2, ...]] where each inner
    list is a flat sequence of vertex coordinates for one contour.
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        # Require contour to have at least 3 points (minimum for a polygon)
        if contour.shape[0] < 3:
            continue
        # Simplify slightly to reduce JSON size without losing meaningful detail
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx  = cv2.approxPolyDP(contour, epsilon, True)
        # Flatten to [x1, y1, x2, y2, ...] and convert to Python ints
        flat = approx.flatten().tolist()
        if len(flat) >= 6:   # at least 3 coordinate pairs
            polygons.append(flat)

    return polygons


# ---------------------------------------------------------------------------
# Save mask as PNG for visual QC
# ---------------------------------------------------------------------------

def save_mask_png(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(str(out_path))


# ---------------------------------------------------------------------------
# Main batch annotation
# ---------------------------------------------------------------------------

def annotate_all(
    std_dir: Path,
    ann_dir: Path,
    reports_dir: Path,
    model_id: str = "facebook/sam2-hiera-large",
    dry_run: bool = False,
    save_masks: bool = True,
) -> None:

    logger.info("=" * 70)
    logger.info("SAM 2 AUTOMATED ANNOTATION  --  surgeonfish dataset")
    logger.info("=" * 70)
    logger.info(f"Input:    {std_dir}")
    logger.info(f"Output:   {ann_dir}")
    logger.info(f"Model:    {model_id}")
    logger.info(f"Dry run:  {dry_run}")
    logger.info("")

    # Discover images
    all_images = []
    for genus_dir in sorted(std_dir.iterdir()):
        if not genus_dir.is_dir():
            continue
        imgs = sorted([f for f in genus_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXT])
        if imgs:
            logger.info(f"  {genus_dir.name}: {len(imgs)} images")
        all_images.extend(imgs)

    logger.info(f"\nTotal images to annotate: {len(all_images)}\n")

    if dry_run:
        logger.info("Dry run -- no SAM inference or file writes.")
        for p in all_images:
            logger.info(f"  would annotate: {p.parent.name}/{p.name}")
        return

    # Load SAM 2
    predictor, device = load_sam2(model_id)

    # Setup output directories
    ann_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = ann_dir / "masks"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # COCO JSON skeleton
    coco = {
        "info": {
            "description": "Surgeonfish (Acanthuridae) segmentation dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "fish", "supercategory": "animal"}
        ],
        "images": [],
        "annotations": [],
    }

    log_rows  = []
    image_id  = 0
    ann_id    = 0
    n_ok      = 0
    n_low     = 0
    n_failed  = 0

    for img_path in all_images:
        image_id += 1
        genus   = img_path.parent.name
        species = img_path.stem

        logger.info(f"[{image_id:03d}/{len(all_images):03d}] {genus}/{img_path.name}")

        # Image size (should always be 1024x1024 after standardization)
        with Image.open(img_path) as pil_img:
            W, H = pil_img.size

        # Add image entry regardless of annotation outcome
        coco["images"].append({
            "id":        image_id,
            "file_name": f"{genus}/{img_path.name}",
            "width":     W,
            "height":    H,
            "genus":     genus,
            "species":   species,
        })

        # Run SAM 2
        result = annotate_image(predictor, img_path, device)

        log_entry = {
            "image_id":      image_id,
            "genus":         genus,
            "species":       species,
            "file_name":     img_path.name,
            "status":        result["status"],
            "sam_score":     round(result["score"], 4),
            "mask_area_frac":round(result["mask_area_frac"], 4),
            "strategy":      result["strategy"],
            "message":       result["message"],
        }
        log_rows.append(log_entry)

        if result["status"] == "OK":
            n_ok += 1
            mask = result["mask"]

            # Save mask PNG for QC
            if save_masks:
                mask_path = mask_dir / genus / img_path.with_suffix(".png").name
                save_mask_png(mask, mask_path)

            # Convert mask to COCO polygon
            polygons = mask_to_coco_polygon(mask)
            if not polygons:
                logger.warning(f"  No polygon contour found -- skipping annotation")
                continue

            ann_id += 1
            x, y, bw, bh = result["bbox"]

            coco["annotations"].append({
                "id":           ann_id,
                "image_id":     image_id,
                "category_id":  1,
                "segmentation": polygons,
                "area":         float(mask.sum()),
                "bbox":         [x, y, bw, bh],
                "iscrowd":      0,
            })
            logger.info(f"  [OK]  score={result['score']:.3f}  "
                        f"area={result['mask_area_frac']:.1%}  "
                        f"strategy={result['strategy']}")

        elif result["status"] == "LOW_CONFIDENCE":
            n_low += 1
            logger.warning(f"  [LOW] {result['message']}")
            # Still save the mask for manual inspection
            if save_masks and result["mask"] is not None:
                mask_path = mask_dir / genus / ("LOW_" + img_path.with_suffix(".png").name)
                save_mask_png(result["mask"], mask_path)
        else:
            n_failed += 1
            logger.error(f"  [FAIL] {result['message']}")

    # Write COCO JSON
    ann_path = ann_dir / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=2)
    logger.info(f"\nAnnotations saved: {ann_path}")

    # Write CSV log
    log_path = reports_dir / "annotation_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    logger.info(f"Log saved: {log_path}")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("ANNOTATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total images:         {len(all_images)}")
    logger.info(f"  Annotated (OK):     {n_ok}")
    logger.info(f"  Low confidence:     {n_low}  (inspect masks/LOW_*.png)")
    logger.info(f"  Failed:             {n_failed}")
    logger.info(f"COCO annotations:     {len(coco['annotations'])}")
    logger.info("")
    if n_low > 0:
        logger.info("Low-confidence images (review manually):")
        for row in log_rows:
            if row["status"] == "LOW_CONFIDENCE":
                logger.info(f"  {row['genus']}/{row['file_name']}  -- {row['message']}")
    logger.info("")
    logger.info("Next step:")
    logger.info("  Review masks in data/annotations/masks/ visually.")
    logger.info("  If any masks are wrong, re-run with --fix-single or")
    logger.info("  edit annotations.json manually for those images.")
    logger.info("  Then proceed to train_mask_rcnn.py.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automated SAM 2 annotation for Mask R-CNN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/python/generate_annotations.py
  python scripts/python/generate_annotations.py --dry-run
  python scripts/python/generate_annotations.py --model facebook/sam2-hiera-small
  python scripts/python/generate_annotations.py --single data/standardized_images/Acanthurus/Acanthurus_mata.png
        """,
    )
    parser.add_argument("--input",   type=Path, default=STD_DIR,
                        help="Standardized images root (default: data/standardized_images)")
    parser.add_argument("--output",  type=Path, default=ANN_DIR,
                        help="Annotation output dir (default: data/annotations)")
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR,
                        help="Reports directory (default: reports/)")
    parser.add_argument("--model",   type=str,  default="facebook/sam2-hiera-large",
                        help="SAM 2 model ID from HuggingFace (default: sam2-hiera-large)")
    parser.add_argument("--no-masks", action="store_true",
                        help="Skip saving mask PNGs (faster, no QC images)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List images without running SAM")
    parser.add_argument("--single",  type=Path, default=None,
                        help="Annotate a single image and print result")

    args = parser.parse_args()

    if args.single:
        # Single-image mode: load SAM, annotate, show result
        predictor, device = load_sam2(args.model)
        result = annotate_image(predictor, args.single, device)
        print(f"\nStatus:        {result['status']}")
        print(f"SAM score:     {result['score']:.4f}")
        print(f"Mask area:     {result['mask_area_frac']:.1%}")
        print(f"Strategy:      {result['strategy']}")
        print(f"Bbox [x,y,w,h]:{result['bbox']}")
        print(f"Message:       {result['message']}")

        if result["mask"] is not None:
            out_path = args.output / "masks" / "single_test.png"
            save_mask_png(result["mask"], out_path)
            print(f"Mask saved:    {out_path}")
    else:
        annotate_all(
            std_dir    = args.input,
            ann_dir    = args.output,
            reports_dir= args.reports,
            model_id   = args.model,
            dry_run    = args.dry_run,
            save_masks = not args.no_masks,
        )


if __name__ == "__main__":
    main()
