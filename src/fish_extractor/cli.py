"""Command-line entry point for the fish-identification/extraction pipeline.

No detection/segmentation models are loaded unless --run is passed
(default off) - a plain invocation only does local, no-model operations
(e.g. refreshing the review page, or applying review-page exclusions).

Usage:
    python -m fish_extractor.cli --run
    python -m fish_extractor.cli --run --species "Zebrasoma flavescens"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fish_extractor.config import PipelineConfig
from fish_extractor.pipeline import FishExtractorPipeline
from fish_extractor.review import apply_review_feedback, generate_review_html


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species", action="append", default=None,
        help='Restrict the run to one species (repeatable), e.g. '
             '--species "Zebrasoma flavescens".',
    )
    parser.add_argument(
        "--apply-review", type=Path, default=None,
        help="Path to a fish_extraction_review_feedback.json exported from the review "
             "page - marks the listed images permanently excluded.",
    )
    parser.add_argument(
        "--no-review-page", action="store_true",
        help="Skip regenerating reports/fish_extraction_review.html at the end of this run.",
    )
    parser.add_argument(
        "--run", action="store_true", default=False,
        help="Actually load the detection/segmentation models and process images. "
             "Default: off - without this flag, no model is loaded; --apply-review "
             "still applies exclusions but does not process new images. This default "
             "exists so a plain run (or a future end-to-end pipeline invocation) never "
             "silently loads multi-GB models.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for ``python -m fish_extractor.cli``.

    Dispatches to whichever mode the flags select — applying review feedback,
    generating the review page, or a detection/segmentation run. The GPU
    models are gated behind ``--run``, so a bare invocation stays cheap and
    cannot start downloading checkpoints.
    """
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    config = PipelineConfig()

    if args.apply_review is not None:
        excluded = apply_review_feedback(config, args.apply_review)
        print(f"Applied review feedback: {excluded} image(s) marked permanently excluded.")

    if not args.run:
        print("--run not set: no models loaded, skipping extraction.")
        if not args.no_review_page:
            review_path = generate_review_html(config, config.review_path)
            print(f"Review page refreshed: {review_path.resolve()}")
        return

    pipeline = FishExtractorPipeline(config)
    species_filter = set(args.species) if args.species else None
    results = pipeline.run(species_filter=species_filter)

    accepted = [r for r in results if r.status == "accepted"]
    flagged = [r for r in results if r.status == "flagged"]
    print(f"\n{len(accepted)}/{len(results)} image(s) accepted, {len(flagged)} flagged for review.")

    if not args.no_review_page:
        review_path = generate_review_html(config, config.review_path)
        print(f"\nReview page: {review_path.resolve()}")
        print("Open it in a browser, check exclusions, click 'Export exclusions', then:")
        print(
            "  python -m fish_extractor.cli --apply-review "
            "reports/fish_extraction_review_feedback.json"
        )


if __name__ == "__main__":
    main()
