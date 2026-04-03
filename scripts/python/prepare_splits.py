# -*- coding: utf-8 -*-
"""
prepare_splits.py

Creates train / val / test split files for Mask R-CNN.

No files are copied or moved. The standardized_images directory is used
as-is. Three lightweight text files are written to data/annotations/
listing the image IDs that belong to each split.

The training script reads annotations.json once and filters by these IDs.

Output:
  data/annotations/
    annotations.json   <- already exists (from generate_annotations.py)
    train_ids.txt      <- 43 image IDs, one per line
    val_ids.txt        <- 10 image IDs
    test_ids.txt       <- 11 image IDs  (10 clean + Naso annulatus)
    split_summary.json <- human-readable record of which species is in which split

Split (seed=42, stratified by genus):
  Train  43  Acanthurus x21, Ctenochaetus x7, Naso x9,
             Paracanthurus x1, Prionurus x1, Zebrasoma x4
  Val    10  Acanthurus x5, Ctenochaetus x1, Naso x2,
             Prionurus x1, Zebrasoma x1
  Test   11  Acanthurus x5, Ctenochaetus x1, Naso x3 (incl. annulatus),
             Prionurus x1, Zebrasoma x1

Note on Naso annulatus:
  SAM produced a low-confidence mask so it has no entry in annotations.json.
  It is placed in test so the trained model predicts its mask at inference.

Usage:
  python scripts/python/prepare_splits.py
  python scripts/python/prepare_splits.py --dry-run
"""

import json
import sys
import logging
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
ANN_DIR      = DATA_DIR / "annotations"
ANN_JSON     = ANN_DIR / "annotations.json"

# ---------------------------------------------------------------------------
# Split assignments  (seed=42, stratified by genus)
# ---------------------------------------------------------------------------

TRAIN_SPECIES = [
    "Acanthurus albipectoralis",
    "Acanthurus bariene",
    "Acanthurus chronixis",
    "Acanthurus dussumieri",
    "Acanthurus grammoptilus",
    "Acanthurus guttatus",
    "Acanthurus japonicus",
    "Acanthurus leucocheilus",
    "Acanthurus leucosternon",
    "Acanthurus lineatus",
    "Acanthurus maculiceps",
    "Acanthurus mata",
    "Acanthurus monroviae",
    "Acanthurus nigricans",
    "Acanthurus nigricauda",
    "Acanthurus olivaceus",
    "Acanthurus sohal",
    "Acanthurus tractus",
    "Acanthurus tristis",
    "Acanthurus_achilles",
    "Acanthurus_xanthopterus",
    "Ctenochaetus cyanocheilus",
    "Ctenochaetus flavicauda",
    "Ctenochaetus hawaiiensis",
    "Ctenochaetus striatus",
    "Ctenochaetus strigosus",
    "Ctenochaetus tominiensis",
    "Ctenochaetus truncatus",
    "Naso brachycentron",
    "Naso caesius",
    "Naso hexacanthus",
    "Naso maculatus",
    "Naso minor",
    "Naso tergus",
    "Naso tonganus",
    "Naso unicornis",
    "Naso vlamingii",
    "Paracanthurus_hepatus",
    "Prionurus laticlavius",
    "Zebrasoma flavescens",
    "Zebrasoma gemmatum",
    "Zebrasoma veliferum",
    "Zebrasoma xanthurum",
]

VAL_SPECIES = [
    "Acanthurus chirurgus",
    "Acanthurus fowleri",
    "Acanthurus nubilus",
    "Acanthurus tennentii",
    "Acanthurus_triostegus",
    "Ctenochaetus binotatus",
    "Naso brevirostris",
    "Naso tuberosus",
    "Prionurus chrysurus",
    "Zebrasoma desjardinii",
]

TEST_SPECIES = [
    "Acanthurus albimento",
    "Acanthurus blochii",
    "Acanthurus gahhm",
    "Acanthurus nigrofuscus",
    "Acanthurus pyroferus",
    "Ctenochaetus marginatus",
    "Naso fageni",
    "Naso lituratus",
    "Naso annulatus",
    "Prionurus microlepidotus",
    "Zebrasoma scopas",
]

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
# Main
# ---------------------------------------------------------------------------

def prepare_splits(ann_json_path: Path, ann_dir: Path, dry_run: bool) -> None:

    logger.info("=" * 70)
    logger.info("PREPARING TRAIN / VAL / TEST SPLITS")
    logger.info("=" * 70)
    logger.info(f"annotations.json : {ann_json_path}")
    logger.info(f"Output directory : {ann_dir}")
    logger.info(f"Dry run          : {dry_run}")
    logger.info("")

    if not ann_json_path.exists():
        logger.error(f"annotations.json not found: {ann_json_path}")
        logger.error("Run generate_annotations.py first.")
        sys.exit(1)

    with open(ann_json_path) as f:
        coco = json.load(f)

    # Build lookup: species stem -> image_id
    # file_name in the JSON is "Genus/Species name.png"
    species_to_id = {
        Path(img["file_name"]).stem: img["id"]
        for img in coco["images"]
    }

    logger.info(f"Loaded {len(coco['images'])} images, "
                f"{len(coco['annotations'])} annotations")
    logger.info("")

    splits = {
        "train": TRAIN_SPECIES,
        "val":   VAL_SPECIES,
        "test":  TEST_SPECIES,
    }

    all_ids  = {}
    summary  = {}

    for split_name, species_list in splits.items():
        ids        = []
        annotated  = []
        unannotated= []

        for sp in species_list:
            if sp in species_to_id:
                ids.append(species_to_id[sp])
                annotated.append(sp)
            else:
                unannotated.append(sp)
                logger.warning(
                    f"  [{split_name}] '{sp}' not in annotations.json "
                    f"(no annotation -- inference only)"
                )

        all_ids[split_name] = ids
        summary[split_name] = {
            "annotated":   annotated,
            "unannotated": unannotated,
            "total_images": len(species_list),
            "n_annotated":  len(annotated),
        }

        id_set = set(ids)
        n_anns = sum(1 for a in coco["annotations"] if a["image_id"] in id_set)
        logger.info(f"{split_name.upper():<6}  {len(species_list):2d} images  "
                    f"{n_anns:2d} annotations")
        for sp in species_list:
            tag = "  [no annotation -- inference only]" if sp not in species_to_id else ""
            logger.info(f"  {sp}{tag}")
        logger.info("")

    # Validate no image in more than one split
    all_flat = all_ids["train"] + all_ids["val"] + all_ids["test"]
    if len(all_flat) != len(set(all_flat)):
        logger.error("Duplicate image IDs across splits -- aborting.")
        sys.exit(1)

    # Write split ID files
    if not dry_run:
        ann_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in all_ids.items():
        out_path = ann_dir / f"{split_name}_ids.txt"
        content  = "\n".join(str(i) for i in sorted(ids)) + "\n"
        if not dry_run:
            out_path.write_text(content)
        logger.info(f"{'Would write' if dry_run else 'Written'}: {out_path}  "
                    f"({len(ids)} IDs)")

    # Write split_summary.json
    summary_path = ann_dir / "split_summary.json"
    summary_out  = {
        "counts": {
            k: {
                "total_images": v["total_images"],
                "annotated":    v["n_annotated"],
                "unannotated":  len(v["unannotated"]),
            }
            for k, v in summary.items()
        },
        "splits": summary,
        "notes": {
            "Naso annulatus": (
                "Low-confidence SAM mask. No annotation. "
                "Placed in test -- model predicts mask at inference."
            ),
            "Paracanthurus_hepatus": (
                "Only 1 image for this genus -- train only."
            ),
        },
    }
    if not dry_run:
        with open(summary_path, "w") as f:
            json.dump(summary_out, f, indent=2)
    logger.info(f"{'Would write' if dry_run else 'Written'}: {summary_path}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Train : {len(TRAIN_SPECIES):2d} images  "
                f"({len(all_ids['train'])} annotated)")
    logger.info(f"  Val   : {len(VAL_SPECIES):2d} images  "
                f"({len(all_ids['val'])} annotated)")
    logger.info(f"  Test  : {len(TEST_SPECIES):2d} images  "
                f"({len(all_ids['test'])} annotated + "
                f"{len(TEST_SPECIES) - len(all_ids['test'])} inference-only)")
    logger.info("")
    logger.info("data/annotations/ now contains:")
    logger.info("  annotations.json   <- full dataset, unchanged")
    logger.info("  train_ids.txt      <- 43 image IDs")
    logger.info("  val_ids.txt        <- 10 image IDs")
    logger.info("  test_ids.txt       <- 11 image IDs")
    logger.info("  split_summary.json <- human-readable split record")
    logger.info("")
    logger.info("Next step: python scripts/python/train_mask_rcnn.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test split ID files for Mask R-CNN",
    )
    parser.add_argument(
        "--ann-json", type=Path, default=ANN_JSON,
        help="Path to annotations.json (default: data/annotations/annotations.json)",
    )
    parser.add_argument(
        "--ann-dir", type=Path, default=ANN_DIR,
        help="Output directory for split files (default: data/annotations/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without creating any files",
    )
    args = parser.parse_args()

    prepare_splits(
        ann_json_path=args.ann_json,
        ann_dir=args.ann_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
