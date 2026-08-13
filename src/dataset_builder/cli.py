"""Command-line entry point for the dataset-building pipeline.

No GBIF requests are made unless --scrape-web is passed (default off) - a
plain invocation only does local, no-network operations (e.g. refreshing
the review page, or purging rejects via --apply-review without backfill).

Usage:
    python -m dataset_builder.cli --scrape-web
    python -m dataset_builder.cli --scrape-web --species "Zebrasoma flavescens" \
        --target-per-species 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dataset_builder.archive import unzip_all, zip_all
from dataset_builder.config import PipelineConfig
from dataset_builder.pipeline import DatasetBuilderPipeline
from dataset_builder.renumber import renumber_all
from dataset_builder.review import apply_review_feedback, generate_review_html

_DEFAULT_REVIEW_PATH = Path("reports/review.html")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-per-species", type=int, default=None,
        help="Override the balanced per-species image target (default: 25).",
    )
    parser.add_argument(
        "--species", action="append", default=None,
        help='Restrict the run to one species (repeatable), e.g. --species "Zebrasoma flavescens".',
    )
    parser.add_argument(
        "--apply-review", type=Path, default=None,
        help="Path to a review_feedback.json exported from the review page - "
             "deletes the rejected images and backfills each affected species "
             "before this run's collection pass.",
    )
    parser.add_argument(
        "--no-review-page", action="store_true",
        help="Skip regenerating reports/review.html at the end of this run.",
    )
    parser.add_argument(
        "--renumber", action="store_true",
        help="Renumber every species folder's files to a clean 001..N sequence "
             "(fixes duplicate-index filenames from repeated review/backfill "
             "rounds) and exit, skipping collection.",
    )
    parser.add_argument(
        "--zip", action="store_true",
        help="Zip every species folder into <species>.zip, removing the loose "
             "files, and exit, skipping collection.",
    )
    parser.add_argument(
        "--unzip", action="store_true",
        help="Extract every <species>.zip back into a loose-files folder and "
             "exit, skipping collection. Run this before any later phase that "
             "expects data/raw_images/ to contain plain image files.",
    )
    parser.add_argument(
        "--scrape-web", action="store_true", default=False,
        help="Actually query GBIF and download new candidate images for the "
             "current genera/species. Default: off - without this flag, no "
             "network requests are made; --apply-review still purges rejected "
             "images but does not backfill replacements. This default exists "
             "so a plain run (or a future end-to-end pipeline invocation) "
             "never silently triggers a multi-round GBIF fetch.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    config = PipelineConfig()
    if args.target_per_species is not None:
        config.target_per_species = args.target_per_species

    if args.renumber:
        results = renumber_all(config.raw_images_root, config.metadata_csv_path)
        total = sum(len(v) for v in results.values())
        print(f"Renumbered {total} file(s) across {len(results)} species.")
        return

    if args.zip:
        zips = zip_all(config.raw_images_root)
        print(f"Zipped {len(zips)} species folder(s).")
        return

    if args.unzip:
        extracted = unzip_all(config.raw_images_root)
        print(f"Unzipped {len(extracted)} species zip(s).")
        return

    if args.apply_review is not None:
        removed = apply_review_feedback(config, args.apply_review)
        total = sum(removed.values())
        print(f"Applied review feedback: removed {total} image(s) across {len(removed)} species.")

    if not args.scrape_web:
        print("--scrape-web not set: no GBIF requests made, skipping collection.")
        if args.apply_review is not None:
            print("(Rejected images were purged above, but not backfilled.)")
        if not args.no_review_page:
            review_path = generate_review_html(config, _DEFAULT_REVIEW_PATH)
            print(f"Review page refreshed: {review_path.resolve()}")
        return

    pipeline = DatasetBuilderPipeline(config)
    species_filter = set(args.species) if args.species else None
    results = pipeline.run(species_filter=species_filter)

    met = [r for r in results if r.met_target]
    short = [r for r in results if not r.met_target]
    print(f"\n{len(met)}/{len(results)} species reached target.")
    if short:
        print("Short of target:")
        for r in short:
            print(f"  {r.species_name}: {r.accepted_count}/{r.target}")

    if not args.no_review_page:
        review_path = generate_review_html(config, _DEFAULT_REVIEW_PATH)
        print(f"\nReview page: {review_path.resolve()}")
        print("Open it in a browser, uncheck bad images, click 'Export rejections', then:")
        print(
            "  python -m dataset_builder.cli --apply-review reports/review_feedback.json "
            "--scrape-web"
        )


if __name__ == "__main__":
    main()
