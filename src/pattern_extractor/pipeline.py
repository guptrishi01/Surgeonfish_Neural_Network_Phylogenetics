"""Orchestrates the pattern-extraction pipeline end to end.

Walks fish_extractor's output (data/extracted_fish/), fits colour
clustering per species from a reference image, then extracts coloring,
stripe, and spot features for every image of that species and writes one
row per image to a CSV.

Unlike dataset_builder/fish_extractor, this pipeline keeps no resumable
per-image state: extraction here is a fast, deterministic, local
computation (no network or GPU calls), so a re-run is cheap and simply
overwrites the output CSV rather than needing to skip already-processed
images.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from pattern_extractor.clustering import assign_clusters, fit_reference
from pattern_extractor.color import extract_color_features
from pattern_extractor.config import PipelineConfig
from pattern_extractor.spot import extract_spot_features
from pattern_extractor.stripe import extract_stripe_features

logger = logging.getLogger(__name__)

_FIELDNAMES = [
    "image_key", "dominant_fraction", "is_solid", "n_significant_colors",
    "spot_count", "mean_spot_area", "spot_present",
    "elongated_region_count", "periodicity_strength", "stripe_present",
]


def _load_image_and_mask(species_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Loads a fish_extractor output pair, or None if either file is missing."""
    image_path = species_dir / f"{stem}.png"
    mask_path = species_dir / f"{stem}_mask.png"
    if not image_path.exists() or not mask_path.exists():
        return None
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    return image, mask


class PatternExtractorPipeline:
    """Extracts coloring/stripe/spot features for every extracted fish image."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def run(self, species_filter: set[str] | None = None) -> list[dict]:
        """Processes every species under extracted_root and writes the output CSV.

        Args:
            species_filter: If given, restricts the run to these species
                names (``"Genus species"``, space-separated).

        Returns:
            One feature row (as a dict) per successfully processed image.
        """
        rows: list[dict] = []
        root = self._config.extracted_root
        for genus_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for species_dir in sorted(p for p in genus_dir.iterdir() if p.is_dir()):
                species_name = species_dir.name.replace("_", " ")
                if species_filter and species_name not in species_filter:
                    continue
                rows.extend(self._process_species(species_dir))
        self._write_csv(rows)
        return rows

    def _process_species(self, species_dir: Path) -> list[dict]:
        """Fits clustering from this species' reference image, then extracts every image.

        A single image (including the reference) that fails to cluster -
        e.g. an unexpectedly empty mask - is logged and skipped rather than
        aborting the whole species, matching fish_extractor's per-item
        fault isolation.
        """
        stems = sorted(
            p.stem for p in species_dir.glob("*.png") if not p.stem.endswith("_mask")
        )
        if not stems:
            return []

        reference_stem = (
            self._config.reference_filename_stem
            if self._config.reference_filename_stem in stems
            else stems[0]
        )
        reference = _load_image_and_mask(species_dir, reference_stem)
        if reference is None:
            logger.warning("No mask found for reference image in %s, skipping species", species_dir)
            return []
        reference_image, reference_mask = reference
        try:
            reference_centers = fit_reference(
                reference_image, reference_mask, self._config.clustering
            )
        except ValueError:
            logger.exception(
                "Could not fit reference clustering for %s, skipping species", species_dir
            )
            return []

        rows = []
        for stem in stems:
            loaded = _load_image_and_mask(species_dir, stem)
            if loaded is None:
                logger.warning("No mask found for %s/%s, skipping image", species_dir, stem)
                continue
            try:
                row = self._process_image(species_dir, stem, loaded, reference_centers)
            except ValueError:
                logger.exception(
                    "Could not extract features for %s/%s, skipping image", species_dir, stem
                )
                continue
            rows.append(row)
        return rows

    def _process_image(
        self,
        species_dir: Path,
        stem: str,
        loaded: tuple[np.ndarray, np.ndarray],
        reference_centers: np.ndarray,
    ) -> dict:
        """Clusters one image and extracts its color/spot/stripe feature row."""
        image, mask = loaded
        cluster_result = assign_clusters(image, mask, reference_centers, self._config.clustering)
        color = extract_color_features(cluster_result, self._config.color)
        spot = extract_spot_features(cluster_result, self._config.region, self._config.spot)
        stripe = extract_stripe_features(
            image, mask, cluster_result, self._config.region, self._config.stripe
        )

        image_key = str(
            species_dir.relative_to(self._config.extracted_root) / stem
        ).replace("\\", "/")
        return {
            "image_key": image_key,
            "dominant_fraction": color.dominant_fraction,
            "is_solid": color.is_solid,
            "n_significant_colors": color.n_significant_colors,
            "spot_count": spot.spot_count,
            "mean_spot_area": spot.mean_spot_area,
            "spot_present": spot.spot_present,
            "elongated_region_count": stripe.elongated_region_count,
            "periodicity_strength": stripe.periodicity_strength,
            "stripe_present": stripe.stripe_present,
        }

    def _write_csv(self, rows: list[dict]) -> None:
        """Overwrites output_csv_path with one row per successfully processed image."""
        path = self._config.output_csv_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote %d pattern-feature row(s) to %s", len(rows), path)
