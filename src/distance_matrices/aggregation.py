"""Collapses pattern_extractor's per-image feature rows into one vector per species.

See the package docstring (``__init__.py``) for the aggregation choices
(reference-image policy, plain arithmetic means, spot-area normalization)
and their reasoning.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from distance_matrices.config import AggregationConfig
from distance_matrices.species_coverage import load_matched_species

logger = logging.getLogger(__name__)

_OUTPUT_FIELDNAMES = [
    "species", "n_images",
    "mean_dominant_fraction", "mean_hue_dispersion", "mean_n_significant_colors", "prop_solid",
    "mean_elongated_region_count", "mean_periodicity_strength", "prop_striped",
    "mean_spot_count", "mean_spot_area_fraction", "prop_spotted",
]


@dataclass
class SpeciesAggregate:
    """One species' aggregated pattern-feature vector, per dimension.

    Attributes:
        species: "Genus_species".
        n_images: Number of images this aggregate was computed from, after
            the reference-image policy and min-image filter were applied -
            recorded alongside every result, not just described as a
            policy, so a downstream reader can check it directly rather
            than trust it silently.
        mean_dominant_fraction: Coloring dimension - mean largest-cluster
            fraction across the species' images.
        mean_hue_dispersion: Coloring dimension - mean hue/saturation
            dispersion (see ``pattern_extractor.color``).
        mean_n_significant_colors: Coloring dimension - mean count of
            clusters covering a non-trivial body share.
        prop_solid: Coloring dimension - fraction of the species' images
            classified ``is_solid``.
        mean_elongated_region_count: Stripe dimension - mean count of
            elongated, narrow candidate-stripe regions.
        mean_periodicity_strength: Stripe dimension - mean FFT
            periodicity signal.
        prop_striped: Stripe dimension - fraction of images classified
            ``stripe_present``. See ``StripeConfig``'s docstring for why
            this should be read as a lower bound, not a precise rate -
            the classifier is calibrated to rarely false-positive but
            frequently misses real stripes (~14% recall).
        mean_spot_count: Spot dimension - mean count of spot-like regions.
        mean_spot_area_fraction: Spot dimension - mean spot area as a
            fraction of each image's own masked-pixel count (normalized
            per image before averaging - see the package docstring).
        prop_spotted: Spot dimension - fraction of images classified
            ``spot_present``.
    """

    species: str
    n_images: int
    mean_dominant_fraction: float
    mean_hue_dispersion: float
    mean_n_significant_colors: float
    prop_solid: float
    mean_elongated_region_count: float
    mean_periodicity_strength: float
    prop_striped: float
    mean_spot_count: float
    mean_spot_area_fraction: float
    prop_spotted: float


def _species_of(image_key: str) -> str:
    return image_key.split("/")[1]


def _total_masked_pixels(extracted_root: Path, image_key: str) -> int:
    """Reads an image's mask file and returns its masked-in (fish) pixel count."""
    mask_path = extracted_root / f"{image_key}_mask.png"
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    return int(mask.sum())


def aggregate_species_features(config: AggregationConfig) -> list[SpeciesAggregate]:
    """Collapses per-image feature rows into one aggregate vector per species.

    Restricted to the species in ``species_coverage_csv_path`` with real
    genetic-data phylogenetic placement (``has_genetic_data == "yes"``) -
    the only ones Phase 4's Kmult test can use. Species with fewer than
    ``config.min_images_per_species`` qualifying images after the
    ``include_reference_images`` policy is applied are dropped and logged,
    not silently included with too little data to be meaningful.

    Args:
        config: Aggregation settings.

    Returns:
        One SpeciesAggregate per species with enough qualifying images,
        sorted by species name - callers needing the count actually used
        (e.g. for an ``assert matched == N`` check) should check
        ``len()`` of the result themselves rather than assume 50, since
        both the reference-image policy and min_images_per_species can
        reduce it.
    """
    matched_species = {
        m.species_key for m in load_matched_species(config.species_coverage_csv_path)
    }

    with open(config.pattern_features_csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_species: dict[str, list[dict]] = {}
    for row in rows:
        species = _species_of(row["image_key"])
        if species not in matched_species:
            continue
        if not config.include_reference_images and row["is_reference"] == "True":
            continue
        by_species.setdefault(species, []).append(row)

    dropped_zero = matched_species - set(by_species)
    if dropped_zero:
        logger.warning(
            "%d matched species have zero qualifying images and are excluded: %s",
            len(dropped_zero), sorted(dropped_zero),
        )

    aggregates = []
    for species in sorted(by_species):
        species_rows = by_species[species]
        if len(species_rows) < config.min_images_per_species:
            logger.info(
                "%s: %d qualifying image(s), below min_images_per_species=%d, excluded",
                species, len(species_rows), config.min_images_per_species,
            )
            continue
        aggregates.append(_aggregate_one_species(species, species_rows, config.extracted_root))

    logger.info(
        "Aggregated %d of %d matched species (min_images_per_species=%d, "
        "include_reference_images=%s)",
        len(aggregates), len(matched_species),
        config.min_images_per_species, config.include_reference_images,
    )
    return aggregates


def _aggregate_one_species(
    species: str, rows: list[dict], extracted_root: Path
) -> SpeciesAggregate:
    dominant_fractions = [float(r["dominant_fraction"]) for r in rows]
    hue_dispersions = [float(r["hue_dispersion"]) for r in rows]
    n_significant_colors = [float(r["n_significant_colors"]) for r in rows]
    is_solid = [r["is_solid"] == "True" for r in rows]

    elongated_counts = [float(r["elongated_region_count"]) for r in rows]
    periodicities = [float(r["periodicity_strength"]) for r in rows]
    stripe_present = [r["stripe_present"] == "True" for r in rows]

    spot_counts = [float(r["spot_count"]) for r in rows]
    spot_area_fractions = []
    for r in rows:
        total_masked = _total_masked_pixels(extracted_root, r["image_key"])
        mean_spot_area = float(r["mean_spot_area"])
        spot_area_fractions.append(mean_spot_area / total_masked if total_masked > 0 else 0.0)
    spot_present = [r["spot_present"] == "True" for r in rows]

    return SpeciesAggregate(
        species=species,
        n_images=len(rows),
        mean_dominant_fraction=float(np.mean(dominant_fractions)),
        mean_hue_dispersion=float(np.mean(hue_dispersions)),
        mean_n_significant_colors=float(np.mean(n_significant_colors)),
        prop_solid=float(np.mean(is_solid)),
        mean_elongated_region_count=float(np.mean(elongated_counts)),
        mean_periodicity_strength=float(np.mean(periodicities)),
        prop_striped=float(np.mean(stripe_present)),
        mean_spot_count=float(np.mean(spot_counts)),
        mean_spot_area_fraction=float(np.mean(spot_area_fractions)),
        prop_spotted=float(np.mean(spot_present)),
    )


def write_species_features_csv(aggregates: list[SpeciesAggregate], output_csv_path: Path) -> None:
    """Writes one row per species-aggregate to output_csv_path.

    Args:
        aggregates: Output of aggregate_species_features().
        output_csv_path: Where to write the CSV.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_OUTPUT_FIELDNAMES)
        writer.writeheader()
        for agg in aggregates:
            writer.writerow({name: getattr(agg, name) for name in _OUTPUT_FIELDNAMES})
    logger.info("Wrote %d species aggregate(s) to %s", len(aggregates), output_csv_path)
