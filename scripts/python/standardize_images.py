# -*- coding: utf-8 -*-
"""
standardize_images.py

Standardizes all surgeonfish images in data/raw_images/ for Mask R-CNN input.

What this script does and WHY:
  1. PNG conversion
       JPEG uses lossy compression that introduces blocking artifacts at
       high-frequency regions (fin edges, scale patterns). These artifacts
       corrupt the gradient signal that Mask R-CNN's ResNet backbone uses
       to detect object boundaries. PNG is lossless -- every pixel is
       preserved exactly as captured.

  2. RGB channel enforcement
       Pretrained Mask R-CNN weights (COCO) were trained on 3-channel RGB
       images. Any RGBA image (PNG with transparency) or grayscale image
       must be converted to 3-channel RGB or the model will error or produce
       garbage outputs.

  3. Resize to fixed longest edge (default: 1024 px)
       The standard Mask R-CNN configuration (detectron2, matterport) resizes
       input so the longest edge is 1024 px during training. Feeding wildly
       different resolutions (e.g. 400 px vs 3000 px) causes inconsistent
       feature pyramid scales, degrading mask quality. We match the expected
       training resolution.
       
       Shorter edge is scaled proportionally -- aspect ratio is PRESERVED.
       We do NOT stretch or crop the image; that would distort the fish's
       body proportions, which are biologically meaningful features.

  4. Padding to square (optional, default: on)
       Mask R-CNN batches images together during training. PyTorch requires
       all images in a batch to have the same dimensions. Padding to a
       consistent square (longest_edge x longest_edge) with a mid-grey fill
       (127, 127, 127) avoids shape mismatches without distortion.
       
       Mid-grey (127) is used because it sits at the midpoint of the 0-255
       range and is unlikely to be confused with either the fish body or a
       dark/bright background, minimising any influence of padding on the
       learned features.

  5. uint8 encoding
       Pretrained COCO weights expect pixel values in [0, 255] as uint8.
       OpenCV and PIL may internally use float32 after operations -- this
       script explicitly converts back to uint8 before saving.

  6. Filename cleaning
       Several source filenames contain trailing underscores and mixed case
       (e.g. "Acanthurus_bariene_.jpeg"). These are normalised to clean
       "Genus_species.png" format to prevent downstream path errors and
       ensure the species name parsed from the filename is always correct.

  7. Metadata log
       A CSV log is written to data/standardized_images/standardization_log.csv
       recording the original filename, output filename, original dimensions,
       output dimensions, original format, and any warnings for each image.
       This provides a full audit trail for reproducibility.

Usage:
    # Standardize all images (reads raw_images, writes standardized_images)
    python scripts/python/standardize_images.py

    # Preview what would happen without writing any files
    python scripts/python/standardize_images.py --dry-run

    # Override default longest-edge size (default: 1024)
    python scripts/python/standardize_images.py --size 800

    # Skip padding (output will be rectangular, not square)
    python scripts/python/standardize_images.py --no-pad

    # Process a single image
    python scripts/python/standardize_images.py --single data/raw_images/Acanthurus/Acanthurus_mata.jpg
"""

import os
import sys
import csv
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw_images"
OUT_DIR      = DATA_DIR / "standardized_images"
REPORTS_DIR  = PROJECT_ROOT / "reports"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

GENERA = ["Acanthurus", "Ctenochaetus", "Naso", "Paracanthurus", "Prionurus", "Zebrasoma"]

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
# Core standardization function
# ---------------------------------------------------------------------------

def clean_filename(stem: str) -> str:
    """
    Normalise a species filename stem.

    Examples:
        "Acanthurus_bariene_"  ->  "Acanthurus_bariene"
        "acanthurus_mata"      ->  "Acanthurus_mata"
        "Naso_lopezi__"        ->  "Naso_lopezi"
    """
    # Strip trailing underscores and whitespace
    stem = stem.rstrip("_ ")
    # Capitalise only the first character (genus name)
    if stem:
        stem = stem[0].upper() + stem[1:]
    return stem


def standardize_image(
    input_path: Path,
    output_dir: Path,
    longest_edge: int = 1024,
    pad_to_square: bool = True,
    pad_value: Tuple[int, int, int] = (127, 127, 127),
    dry_run: bool = False,
) -> dict:
    """
    Standardize a single image and write it as PNG.

    Returns a dict with processing details for the log.
    """
    log_entry = {
        "original_file":    input_path.name,
        "original_format":  input_path.suffix.lstrip(".").upper(),
        "original_size":    "",
        "original_channels":"",
        "output_file":      "",
        "output_size":      "",
        "resize_scale":     "",
        "pad_applied":      "",
        "warnings":         "",
        "status":           "",
    }

    # ------------------------------------------------------------------
    # Step 1: Load with PIL first (handles exotic formats and EXIF rotation)
    # ------------------------------------------------------------------
    try:
        pil_img = Image.open(input_path)
        # Apply EXIF orientation before anything else
        try:
            from PIL.ExifTags import TAGS
            exif = pil_img._getexif()
            if exif:
                for tag, val in exif.items():
                    if TAGS.get(tag) == "Orientation":
                        rotation_map = {3: 180, 6: 270, 8: 90}
                        if val in rotation_map:
                            pil_img = pil_img.rotate(rotation_map[val], expand=True)
                            log_entry["warnings"] += "EXIF rotation applied; "
        except Exception:
            pass  # No EXIF or not JPEG -- fine
    except Exception as e:
        log_entry["status"] = f"FAILED: cannot open -- {e}"
        return log_entry

    orig_w, orig_h = pil_img.size
    orig_mode      = pil_img.mode
    log_entry["original_size"]     = f"{orig_w}x{orig_h}"
    log_entry["original_channels"] = orig_mode

    # ------------------------------------------------------------------
    # Step 2: Enforce 3-channel RGB
    # ------------------------------------------------------------------
    if pil_img.mode == "RGBA":
        # Flatten alpha channel onto white -- fish images with transparency
        # are typically product shots; white background is neutral.
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg
        log_entry["warnings"] += "RGBA -> RGB (alpha flattened to white); "
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
        log_entry["warnings"] += f"{orig_mode} -> RGB converted; "

    # Work as numpy array from here for precise control
    img = np.array(pil_img, dtype=np.uint8)  # H x W x 3, RGB order

    # ------------------------------------------------------------------
    # Step 3: Resize so the longest edge equals `longest_edge` pixels.
    #         Aspect ratio is preserved.
    # ------------------------------------------------------------------
    h, w = img.shape[:2]
    if max(h, w) != longest_edge:
        scale  = longest_edge / max(h, w)
        new_w  = int(round(w * scale))
        new_h  = int(round(h * scale))
        # Use LANCZOS (high-quality downsampling) via PIL for clean edges
        pil_resized = Image.fromarray(img).resize(
            (new_w, new_h), resample=Image.LANCZOS
        )
        img   = np.array(pil_resized, dtype=np.uint8)
        scale_str = f"{scale:.4f}"
    else:
        new_w, new_h = w, h
        scale_str    = "1.0000 (no resize)"

    log_entry["resize_scale"] = scale_str

    # ------------------------------------------------------------------
    # Step 4: Pad to square (longest_edge x longest_edge)
    #         Mid-grey (127) padding is neutral and won't bias features.
    # ------------------------------------------------------------------
    if pad_to_square:
        canvas = np.full((longest_edge, longest_edge, 3), pad_value, dtype=np.uint8)
        # Centre the image on the canvas
        y_off  = (longest_edge - new_h) // 2
        x_off  = (longest_edge - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = img
        img = canvas
        pad_str = f"padded ({x_off}px L/R, {y_off}px T/B)"
    else:
        pad_str = "none"

    log_entry["pad_applied"] = pad_str
    log_entry["output_size"] = f"{img.shape[1]}x{img.shape[0]}"

    # ------------------------------------------------------------------
    # Step 5: Build clean output filename and write PNG
    # ------------------------------------------------------------------
    clean_stem  = clean_filename(input_path.stem)
    out_filename = clean_stem + ".png"
    log_entry["output_file"] = out_filename

    if not dry_run:
        # Preserve genus subdirectory structure
        genus = input_path.parent.name
        genus_out_dir = output_dir / genus
        genus_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = genus_out_dir / out_filename

        # Save as PNG using PIL (ensures proper sRGB ICC profile handling)
        Image.fromarray(img, mode="RGB").save(str(out_path), format="PNG", optimize=False)

    log_entry["status"] = "OK (dry run)" if dry_run else "OK"
    return log_entry


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

def standardize_all(
    raw_dir: Path,
    out_dir: Path,
    reports_dir: Path,
    longest_edge: int = 1024,
    pad_to_square: bool = True,
    dry_run: bool = False,
) -> None:

    logger.info("=" * 70)
    logger.info("IMAGE STANDARDIZATION FOR MASK R-CNN")
    logger.info("=" * 70)
    logger.info(f"Input:        {raw_dir}")
    logger.info(f"Output:       {out_dir}")
    logger.info(f"Longest edge: {longest_edge} px")
    logger.info(f"Pad to square:{pad_to_square}")
    logger.info(f"Dry run:      {dry_run}")
    logger.info("")

    # Discover all images
    all_images = []
    for genus_dir in sorted(raw_dir.iterdir()):
        if not genus_dir.is_dir():
            continue
        imgs = sorted([
            f for f in genus_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ])
        if imgs:
            logger.info(f"  {genus_dir.name}: {len(imgs)} images")
        all_images.extend(imgs)

    logger.info(f"\nTotal images to standardize: {len(all_images)}\n")

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

    log_rows  = []
    n_ok      = 0
    n_failed  = 0
    n_warned  = 0
    genus_stats = {}

    for img_path in all_images:
        genus = img_path.parent.name
        if genus not in genus_stats:
            genus_stats[genus] = {"total": 0, "ok": 0, "failed": 0}
        genus_stats[genus]["total"] += 1

        logger.info(f"Processing: {genus}/{img_path.name}")
        entry = standardize_image(
            input_path    = img_path,
            output_dir    = out_dir,
            longest_edge  = longest_edge,
            pad_to_square = pad_to_square,
            dry_run       = dry_run,
        )
        entry["genus"] = genus
        log_rows.append(entry)

        if entry["status"].startswith("OK"):
            n_ok += 1
            genus_stats[genus]["ok"] += 1
            warn_str = entry["warnings"].rstrip("; ") if entry["warnings"] else ""
            if warn_str:
                n_warned += 1
                logger.info(f"  [OK]  {entry['original_size']} -> {entry['output_size']}  |  {warn_str}")
            else:
                logger.info(f"  [OK]  {entry['original_size']} -> {entry['output_size']}")
        else:
            n_failed += 1
            genus_stats[genus]["failed"] += 1
            logger.error(f"  [FAIL] {entry['status']}")

    # ------------------------------------------------------------------
    # Write CSV log
    # ------------------------------------------------------------------
    if not dry_run:
        log_path = reports_dir / "standardization_log.csv"
        fieldnames = [
            "genus", "original_file", "original_format", "original_size",
            "original_channels", "output_file", "output_size", "resize_scale",
            "pad_applied", "warnings", "status",
        ]
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)
        logger.info(f"\nLog saved: {log_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("STANDARDIZATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total processed: {len(all_images)}")
    logger.info(f"  Successful:    {n_ok}  ({n_warned} with warnings)")
    logger.info(f"  Failed:        {n_failed}")
    logger.info("")
    logger.info("Per genus:")
    for genus, stats in sorted(genus_stats.items()):
        logger.info(
            f"  {genus:20s} | total: {stats['total']:3d} | "
            f"ok: {stats['ok']:3d} | failed: {stats['failed']:3d}"
        )
    logger.info("")
    logger.info("What was applied to every image:")
    logger.info("  1. EXIF rotation correction (if present)")
    logger.info("  2. Channel enforcement -> 3-channel RGB uint8")
    logger.info(f"  3. Resize -> longest edge = {longest_edge} px (aspect ratio preserved)")
    if pad_to_square:
        logger.info(f"  4. Padding -> {longest_edge}x{longest_edge} square (mid-grey fill)")
    logger.info("  5. Format -> PNG (lossless)")
    logger.info("  6. Filename -> clean Genus_species.png (trailing underscores removed)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standardize surgeonfish images for Mask R-CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/python/standardize_images.py
  python scripts/python/standardize_images.py --dry-run
  python scripts/python/standardize_images.py --size 800 --no-pad
  python scripts/python/standardize_images.py --single data/raw_images/Acanthurus/Acanthurus_mata.jpg
        """,
    )
    parser.add_argument("--input",   type=Path, default=RAW_DIR,
                        help="Root directory with genus subdirectories (default: data/raw_images)")
    parser.add_argument("--output",  type=Path, default=OUT_DIR,
                        help="Output directory (default: data/standardized_images)")
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR,
                        help="Directory for the CSV log (default: reports/)")
    parser.add_argument("--size",    type=int,  default=1024,
                        help="Longest edge in pixels after resize (default: 1024)")
    parser.add_argument("--no-pad",  action="store_true",
                        help="Skip square padding (output will be rectangular)")
    parser.add_argument("--single",  type=Path, default=None,
                        help="Standardize a single image and print result")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without writing files")

    args = parser.parse_args()

    if args.single:
        entry = standardize_image(
            input_path    = args.single,
            output_dir    = args.output,
            longest_edge  = args.size,
            pad_to_square = not args.no_pad,
            dry_run       = args.dry_run,
        )
        for k, v in entry.items():
            print(f"  {k:<22}: {v}")
    else:
        standardize_all(
            raw_dir       = args.input,
            out_dir       = args.output,
            reports_dir   = args.reports,
            longest_edge  = args.size,
            pad_to_square = not args.no_pad,
            dry_run       = args.dry_run,
        )


if __name__ == "__main__":
    main()
