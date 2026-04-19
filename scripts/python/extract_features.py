# -*- coding: utf-8 -*-
"""
extract_features.py

Extracts a 99-dimensional color and pattern feature vector from each
masked surgeonfish image for downstream phylogenetic comparison.

All features are extracted ONLY from fish-body pixels (inside the mask),
so background coral, water, and sand are completely excluded.

Feature vector (99 dimensions total):
  [0:18]   Hue histogram (18 bins, 0-180 in OpenCV HSV)
            Captures the dominant colors: yellow, blue, brown, green etc.
            Each bin covers a 10-degree hue range.

  [18:26]  Saturation histogram (8 bins)
            Captures color richness: uniform grey fish vs vivid patterned fish.

  [26:34]  Value histogram (8 bins)
            Captures brightness distribution: dark fish vs pale fish.

  [34:54]  Dominant color clusters (5 k-means centers x 3 HSV + 5 frequencies)
            Identifies up to 5 discrete color zones on the body and how much
            of the fish each covers. Captures two-tone, striped, and spotted
            patterns as distinct dominant color areas.

  [54:63]  Dorsal/ventral color gradient (3 regions x 3 HSV channels)
            Mean HSV of dorsal half, ventral half, and their difference.
            Many surgeonfish have countershading (dark dorsal, pale ventral)
            which is a phylogenetically informative trait.

  [63:87]  Gabor texture energy (4 orientations x 3 frequencies x 2 stats)
            Gabor filters detect oriented textures at specific spatial frequencies.
            - Orientations: 0, 45, 90, 135 degrees
            - Frequencies: 0.1 (broad stripes), 0.2 (medium scales), 0.4 (fine dots)
            - Stats: mean and std of filter response energy across fish pixels
            Stripes produce strong 90-degree response; spots produce isotropic
            responses; scale patterns produce high-frequency responses.

  [87:97]  Local Binary Pattern histogram (10 bins, uniform patterns)
            LBP characterizes micro-texture around each pixel. It captures
            whether scales, spots, or fine line patterns are present and
            how densely they occur, independent of color.

  [97:99]  Pattern entropy (2 values)
            Shannon entropy of the hue histogram and LBP histogram.
            High entropy = complex multi-color pattern (e.g. Zebrasoma desjardinii).
            Low entropy = uniform coloration (e.g. Naso hexacanthus).

Outputs:
  data/features/
    features.csv            <- species x 99 feature matrix, with headers
    features.json           <- same data with metadata (genus, species, mask_path)
    feature_names.txt       <- one feature name per line (for reference)
    extraction_log.csv      <- per-image status, mask area, warnings

Usage:
  python scripts/python/extract_features.py

  # Dry run -- list images without extracting
  python scripts/python/extract_features.py --dry-run

  # Extract from a single image (for testing)
  python scripts/python/extract_features.py \
      --single data/standardized_images/Acanthurus/Acanthurus_mata.png \
      --mask outputs/test_predictions/Acanthurus_mata_mask.png

Dependencies:
  pip install scikit-image scipy scikit-learn
  (numpy, opencv, PIL already installed)
"""

import csv
import json
import logging
import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.stats import entropy as scipy_entropy
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from sklearn.cluster import MiniBatchKMeans

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
ANN_DIR      = DATA_DIR / "annotations"
STD_DIR      = DATA_DIR / "standardized_images"
OUT_DIR      = PROJECT_ROOT / "outputs"
FEAT_DIR     = DATA_DIR / "features"

ANN_JSON     = ANN_DIR / "annotations.json"

# Where Mask R-CNN saved per-image binary mask PNGs
# The script checks both test_predictions/ and val_predictions/ and
# predictions/ (single-image predict mode), using whichever exists.
# Search order matters -- all_predictions/ is first because it contains
# masks for ALL 64 species generated with a permissive threshold (0.1).
# test_predictions/ and val_predictions/ are checked as fallbacks only for
# the 21 val/test species where those files exist and may be higher quality.
MASK_SEARCH_DIRS = [
    OUT_DIR / "all_predictions",
    OUT_DIR / "test_predictions",
    OUT_DIR / "val_predictions",
    OUT_DIR / "predictions",
]

GENERA = ["Acanthurus", "Ctenochaetus", "Naso",
          "Paracanthurus", "Prionurus", "Zebrasoma"]

# ---------------------------------------------------------------------------
# Gabor filter parameters
# ---------------------------------------------------------------------------

GABOR_THETAS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GABOR_FREQS  = [0.1, 0.2, 0.4]

# LBP parameters
LBP_RADIUS   = 1
LBP_POINTS   = 8
LBP_N_BINS   = 10

# K-means dominant colors
N_COLORS     = 5

# Feature vector length
N_FEATURES   = 99

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
# Feature name list (for CSV headers and reference)
# ---------------------------------------------------------------------------

def build_feature_names() -> list:
    names = []

    for i in range(18):
        lo = i * 10
        names.append(f"hue_bin_{lo:03d}_{lo+10:03d}")

    for i in range(8):
        lo = i * 32
        names.append(f"sat_bin_{lo:03d}_{lo+32:03d}")

    for i in range(8):
        lo = i * 32
        names.append(f"val_bin_{lo:03d}_{lo+32:03d}")

    for k in range(N_COLORS):
        names += [f"domcol_{k}_hue", f"domcol_{k}_sat", f"domcol_{k}_val"]
    for k in range(N_COLORS):
        names.append(f"domcol_{k}_freq")

    for region in ["dorsal", "ventral", "dv_diff"]:
        for ch in ["hue", "sat", "val"]:
            names.append(f"{region}_{ch}_mean")

    for theta_idx, theta in enumerate(GABOR_THETAS):
        deg = int(np.degrees(theta))
        for freq in GABOR_FREQS:
            fstr = str(freq).replace(".", "p")
            names.append(f"gabor_t{deg:03d}_f{fstr}_mean")
            names.append(f"gabor_t{deg:03d}_f{fstr}_std")

    for i in range(LBP_N_BINS):
        names.append(f"lbp_bin_{i:02d}")

    names += ["hue_entropy", "lbp_entropy"]

    assert len(names) == N_FEATURES, f"Expected {N_FEATURES}, got {len(names)}"
    return names


FEATURE_NAMES = build_feature_names()


# ---------------------------------------------------------------------------
# Core feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    image_path: Path,
    mask_path: Path,
) -> tuple:
    """
    Extract the 99-dimensional feature vector from one fish image+mask pair.

    Returns (feature_vector, log_dict) where feature_vector is a numpy
    array of length N_FEATURES, or (None, log_dict) on failure.
    """
    log = {
        "image_path":  str(image_path),
        "mask_path":   str(mask_path),
        "mask_area_frac": 0.0,
        "n_fish_pixels":  0,
        "status":  "FAILED",
        "warning": "",
    }

    # ------------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------------
    try:
        pil_img = Image.open(image_path).convert("RGB")
        img_np  = np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        log["warning"] = f"Cannot load image: {e}"
        return None, log

    H, W = img_np.shape[:2]

    # ------------------------------------------------------------------
    # Load mask and binarise
    # ------------------------------------------------------------------
    try:
        pil_mask = Image.open(mask_path)
        # Handle both L (grayscale, Mask R-CNN output) and
        # RGB (SAM QC masks saved as 3-channel) formats.
        if pil_mask.mode == "RGB":
            arr = np.array(pil_mask, dtype=np.uint8)
            # If all channels identical it is a binary mask saved as RGB
            if np.allclose(arr[:, :, 0], arr[:, :, 1]) and \
               np.allclose(arr[:, :, 0], arr[:, :, 2]):
                mask_np = arr[:, :, 0]
            else:
                mask_np = np.array(pil_mask.convert("L"), dtype=np.uint8)
        elif pil_mask.mode == "RGBA":
            mask_np = np.array(pil_mask.convert("L"), dtype=np.uint8)
        else:
            mask_np = np.array(pil_mask, dtype=np.uint8)
        # Resize mask to image dimensions if needed
        if pil_mask.size != (W, H):
            pil_mask_l = Image.fromarray(mask_np).resize((W, H), Image.NEAREST)
            mask_np = np.array(pil_mask_l, dtype=np.uint8)
        mask_bin = mask_np > 127
    except Exception as e:
        log["warning"] = f"Cannot load mask: {e}"
        return None, log

    n_fish = int(mask_bin.sum())
    log["n_fish_pixels"]  = n_fish
    log["mask_area_frac"] = round(n_fish / (H * W), 4)

    if n_fish < 500:
        log["warning"] = f"Mask too small ({n_fish} px) -- skipping"
        return None, log

    # ------------------------------------------------------------------
    # Convert to HSV (OpenCV: H=0-180, S=0-255, V=0-255)
    # ------------------------------------------------------------------
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    masked_hsv = hsv[mask_bin]        # shape [N, 3]

    # ------------------------------------------------------------------
    # 1. HSV histograms
    # ------------------------------------------------------------------
    h_hist, _ = np.histogram(
        masked_hsv[:, 0], bins=18, range=(0, 180), density=True
    )
    s_hist, _ = np.histogram(
        masked_hsv[:, 1], bins=8, range=(0, 256), density=True
    )
    v_hist, _ = np.histogram(
        masked_hsv[:, 2], bins=8, range=(0, 256), density=True
    )

    # ------------------------------------------------------------------
    # 2. Dominant color clusters (k-means in HSV space)
    # ------------------------------------------------------------------
    # Normalise HSV to [0, 1] for k-means
    pixels_norm = masked_hsv.astype(np.float32) / np.array([180.0, 255.0, 255.0])

    # Subsample if too many pixels (speed)
    if len(pixels_norm) > 50_000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(pixels_norm), 50_000, replace=False)
        pixels_norm_sub = pixels_norm[idx]
    else:
        pixels_norm_sub = pixels_norm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = MiniBatchKMeans(
            n_clusters=N_COLORS, random_state=42,
            n_init=5, max_iter=200
        )
        km.fit(pixels_norm_sub)

    # Re-predict on all pixels for accurate frequencies
    labels = km.predict(pixels_norm)
    centers = km.cluster_centers_             # [5, 3] normalised HSV
    counts  = np.bincount(labels, minlength=N_COLORS).astype(np.float32)
    freqs   = counts / counts.sum()

    # Sort by frequency (most dominant first) for consistency
    order   = np.argsort(-freqs)
    centers = centers[order]
    freqs   = freqs[order]

    dom_feat = np.concatenate([centers.flatten(), freqs])  # length 20

    # ------------------------------------------------------------------
    # 3. Dorsal/ventral color gradient
    # ------------------------------------------------------------------
    ys, _ = np.where(mask_bin)
    if len(ys) == 0:
        dv_feat = np.zeros(9)
    else:
        y_mid   = (int(ys.min()) + int(ys.max())) / 2

        dorsal_mask  = mask_bin.copy()
        ventral_mask = mask_bin.copy()
        dorsal_mask[int(y_mid):, :]  = False   # keep top half
        ventral_mask[:int(y_mid), :] = False   # keep bottom half

        def mean_hsv_norm(region_mask):
            px = hsv[region_mask]
            if len(px) == 0:
                return np.zeros(3, dtype=np.float32)
            return (px.mean(axis=0) / np.array([180.0, 255.0, 255.0])).astype(np.float32)

        d_mean  = mean_hsv_norm(dorsal_mask)
        v_mean  = mean_hsv_norm(ventral_mask)
        dv_diff = d_mean - v_mean
        dv_feat = np.concatenate([d_mean, v_mean, dv_diff])   # length 9

    # ------------------------------------------------------------------
    # 4. Gabor texture energy
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    gabor_feats = []
    for theta in GABOR_THETAS:
        for freq in GABOR_FREQS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                real, imag = gabor(gray, frequency=freq, theta=theta)
            energy = np.sqrt(real ** 2 + imag ** 2)
            fish_energy = energy[mask_bin]
            gabor_feats.append(float(fish_energy.mean()))
            gabor_feats.append(float(fish_energy.std()))

    gabor_feat = np.array(gabor_feats, dtype=np.float32)   # length 24

    # ------------------------------------------------------------------
    # 5. Local Binary Pattern histogram
    # ------------------------------------------------------------------
    # Convert to uint8 gray (LBP works best on integer images)
    gray_uint8 = (gray * 255).astype(np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lbp = local_binary_pattern(
            gray_uint8, P=LBP_POINTS, R=LBP_RADIUS, method="uniform"
        )

    lbp_hist, _ = np.histogram(
        lbp[mask_bin], bins=LBP_N_BINS,
        range=(0, LBP_POINTS + 2), density=True
    )
    lbp_feat = lbp_hist.astype(np.float32)   # length 10

    # ------------------------------------------------------------------
    # 6. Pattern entropy
    # ------------------------------------------------------------------
    h_norm      = h_hist / (h_hist.sum() + 1e-8)
    lbp_norm    = lbp_hist / (lbp_hist.sum() + 1e-8)
    hue_entropy = float(scipy_entropy(h_norm + 1e-8))
    lbp_entropy = float(scipy_entropy(lbp_norm + 1e-8))
    entropy_feat= np.array([hue_entropy, lbp_entropy], dtype=np.float32)

    # ------------------------------------------------------------------
    # Concatenate
    # ------------------------------------------------------------------
    feat = np.concatenate([
        h_hist.astype(np.float32),
        s_hist.astype(np.float32),
        v_hist.astype(np.float32),
        dom_feat.astype(np.float32),
        dv_feat.astype(np.float32),
        gabor_feat,
        lbp_feat,
        entropy_feat,
    ])

    assert len(feat) == N_FEATURES, f"Feature length mismatch: {len(feat)}"

    # Replace any NaN/Inf with 0
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    log["status"] = "OK"
    return feat, log


# ---------------------------------------------------------------------------
# Mask path resolver
# ---------------------------------------------------------------------------

def find_mask(species_stem: str) -> Path | None:
    """
    Search MASK_SEARCH_DIRS for a mask PNG matching the species stem.
    Returns the first match found, or None.

    The mask filename convention from train_mask_rcnn.py is:
      <species_stem>_mask.png
    """
    for search_dir in MASK_SEARCH_DIRS:
        candidate = search_dir / f"{species_stem}_mask.png"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_all(dry_run: bool = False) -> None:

    logger.info("=" * 70)
    logger.info("FEATURE EXTRACTION  --  color and pattern features")
    logger.info("=" * 70)
    logger.info(f"Images:   {STD_DIR}")
    logger.info(f"Masks:    {[str(d) for d in MASK_SEARCH_DIRS]}")
    logger.info(f"Output:   {FEAT_DIR}")
    logger.info(f"Dry run:  {dry_run}")
    logger.info("")

    # Load annotations to get full species list and file paths
    with open(ANN_JSON) as f:
        coco = json.load(f)

    # Build list of (image_path, species_stem, genus)
    all_items = []
    for img_entry in coco["images"]:
        # file_name is "Genus/Species name.png"
        img_path     = STD_DIR / img_entry["file_name"]
        species_stem = Path(img_entry["file_name"]).stem
        genus        = Path(img_entry["file_name"]).parts[0]
        all_items.append((img_path, species_stem, genus))

    all_items.sort(key=lambda x: x[2] + x[1])   # sort by genus then species
    logger.info(f"Total species: {len(all_items)}")

    # Check mask availability
    found_masks  = 0
    for _, stem, _ in all_items:
        if find_mask(stem) is not None:
            found_masks += 1
    logger.info(f"Masks found:   {found_masks}/{len(all_items)}")

    if found_masks == 0:
        logger.error(
            "No masks found. Run train_mask_rcnn.py --mode test first,\n"
            "then run --mode predict for any remaining images."
        )
        sys.exit(1)

    if dry_run:
        logger.info("\nDry run -- listing images and mask status:")
        for img_path, stem, genus in all_items:
            mask = find_mask(stem)
            status = "MASK OK" if mask else "NO MASK"
            logger.info(f"  [{status}] {genus}/{stem}")
        return

    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    feature_rows = []   # list of (metadata_dict, feat_array)
    log_rows     = []

    for img_path, species_stem, genus in all_items:
        mask_path = find_mask(species_stem)

        if mask_path is None:
            logger.warning(f"  [NO MASK] {genus}/{species_stem} -- skipping")
            log_rows.append({
                "genus": genus, "species": species_stem,
                "image_path": str(img_path), "mask_path": "",
                "mask_area_frac": 0.0, "n_fish_pixels": 0,
                "status": "NO_MASK", "warning": "mask file not found",
            })
            continue

        logger.info(f"  Extracting: {genus}/{species_stem}")
        feat, log = extract_features(img_path, mask_path)

        log["genus"]   = genus
        log["species"] = species_stem
        log_rows.append(log)

        if feat is not None:
            feature_rows.append({
                "genus":      genus,
                "species":    species_stem,
                "image_path": str(img_path),
                "mask_path":  str(mask_path),
                "features":   feat,
            })
            logger.info(
                f"    OK  mask_area={log['mask_area_frac']:.1%}  "
                f"pixels={log['n_fish_pixels']:,}"
            )
        else:
            logger.warning(f"    FAILED: {log['warning']}")

    if not feature_rows:
        logger.error("No features extracted. Check mask paths and image files.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Write features.csv
    # ------------------------------------------------------------------
    csv_path = FEAT_DIR / "features.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["genus", "species"] + FEATURE_NAMES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in feature_rows:
            out = {"genus": row["genus"], "species": row["species"]}
            for name, val in zip(FEATURE_NAMES, row["features"]):
                out[name] = round(float(val), 6)
            writer.writerow(out)
    logger.info(f"\nSaved: {csv_path}  ({len(feature_rows)} species x {N_FEATURES} features)")

    # ------------------------------------------------------------------
    # Write features.json
    # ------------------------------------------------------------------
    json_path = FEAT_DIR / "features.json"
    json_out  = {
        "n_species":   len(feature_rows),
        "n_features":  N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "species": [
            {
                "genus":      r["genus"],
                "species":    r["species"],
                "image_path": r["image_path"],
                "mask_path":  r["mask_path"],
                "features":   [round(float(v), 6) for v in r["features"]],
            }
            for r in feature_rows
        ],
    }
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    logger.info(f"Saved: {json_path}")

    # ------------------------------------------------------------------
    # Write feature_names.txt
    # ------------------------------------------------------------------
    names_path = FEAT_DIR / "feature_names.txt"
    names_path.write_text("\n".join(FEATURE_NAMES) + "\n")
    logger.info(f"Saved: {names_path}")

    # ------------------------------------------------------------------
    # Write extraction log
    # ------------------------------------------------------------------
    log_path = FEAT_DIR / "extraction_log.csv"
    with open(log_path, "w", newline="") as f:
        fieldnames = ["genus", "species", "image_path", "mask_path",
                      "mask_area_frac", "n_fish_pixels", "status", "warning"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(log_rows)
    logger.info(f"Saved: {log_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_ok     = sum(1 for r in log_rows if r["status"] == "OK")
    n_failed = len(log_rows) - n_ok

    logger.info("")
    logger.info("=" * 70)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Processed:   {len(log_rows)}")
    logger.info(f"  Successful:  {n_ok}")
    logger.info(f"  Failed:      {n_failed}")
    logger.info(f"  Feature dim: {N_FEATURES}")
    logger.info("")
    logger.info("Feature groups:")
    logger.info("  Hue histogram      18  dominant colors present on fish body")
    logger.info("  Saturation hist     8  color richness / vividness")
    logger.info("  Value hist          8  brightness distribution")
    logger.info("  Dominant colors    20  up to 5 discrete color zones + coverage")
    logger.info("  Dorsal/ventral      9  countershading and body color gradient")
    logger.info("  Gabor texture      24  stripes / spots / scales at 4 orientations")
    logger.info("  LBP texture        10  micro-texture density and type")
    logger.info("  Pattern entropy     2  overall pattern complexity")
    logger.info("")
    logger.info("Next step:")
    logger.info("  python scripts/python/build_distance_matrix.py")

    if n_failed > 0:
        logger.warning("\nFailed species (no mask or empty mask):")
        for r in log_rows:
            if r["status"] != "OK":
                logger.warning(f"  {r['genus']}/{r['species']}  -- {r['warning']}")


# ---------------------------------------------------------------------------
# Single image mode
# ---------------------------------------------------------------------------

def extract_single(image_path: Path, mask_path: Path) -> None:
    logger.info(f"Extracting features for: {image_path.name}")
    feat, log = extract_features(image_path, mask_path)

    if feat is None:
        logger.error(f"Extraction failed: {log['warning']}")
        sys.exit(1)

    logger.info(f"Status:          {log['status']}")
    logger.info(f"Mask area:       {log['mask_area_frac']:.1%}")
    logger.info(f"Fish pixels:     {log['n_fish_pixels']:,}")
    logger.info(f"Feature vector:  {len(feat)} dimensions")
    logger.info("")

    # Print human-readable summary
    logger.info("Feature summary:")
    logger.info(f"  Dominant hue bins (top 5):")
    hue_bins  = feat[:18]
    top_bins  = np.argsort(-hue_bins)[:5]
    for b in top_bins:
        lo = b * 10
        logger.info(f"    {lo:3d}-{lo+10:3d} deg  weight={hue_bins[b]:.4f}")

    logger.info(f"  Mean saturation: {feat[18:26].mean():.4f}")
    logger.info(f"  Mean value:      {feat[26:34].mean():.4f}")
    logger.info(f"  Pattern entropy (hue):  {feat[97]:.4f}")
    logger.info(f"  Pattern entropy (LBP):  {feat[98]:.4f}")
    logger.info(f"  D/V hue diff:    {feat[60]:.4f}  (+ = dorsal darker hue)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract color and pattern features from masked fish images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List images and mask availability without extracting",
    )
    parser.add_argument(
        "--single", type=Path, default=None,
        help="Extract features from one image (for testing)",
    )
    parser.add_argument(
        "--mask", type=Path, default=None,
        help="Mask PNG for --single mode",
    )
    args = parser.parse_args()

    if args.single:
        if args.mask is None:
            parser.error("--mask is required with --single")
        extract_single(args.single, args.mask)
    else:
        extract_all(dry_run=args.dry_run)


if __name__ == "__main__":
    main()