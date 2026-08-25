"""Unit tests for per-species feature aggregation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from distance_matrices.aggregation import aggregate_species_features
from distance_matrices.config import AggregationConfig

_FEATURE_FIELDNAMES = [
    "image_key", "is_reference", "dominant_fraction", "hue_dispersion", "is_solid",
    "n_significant_colors", "spot_count", "mean_spot_area", "spot_present",
    "elongated_region_count", "periodicity_strength", "stripe_present",
]
_COVERAGE_FIELDNAMES = ["species", "genus", "has_genetic_data", "match_method"]


def _write_features_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FEATURE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_coverage_csv(path: Path, species_list: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COVERAGE_FIELDNAMES)
        writer.writeheader()
        for sp in species_list:
            writer.writerow({
                "species": sp.replace("_", " "), "genus": sp.split("_")[0],
                "has_genetic_data": "yes", "match_method": "exact",
            })


def _write_mask(
    extracted_root: Path, image_key: str, masked_fraction: float, size=(10, 10)
) -> None:
    path = extracted_root / f"{image_key}_mask.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    total = size[0] * size[1]
    n_masked = round(total * masked_fraction)
    mask = np.zeros(total, dtype=np.uint8)
    mask[:n_masked] = 255
    Image.fromarray(mask.reshape(size)).save(path)


def _feature_row(image_key, is_reference=False, dominant_fraction=0.9, hue_dispersion=0.1,
                  is_solid=True, n_significant_colors=1, spot_count=0, mean_spot_area=0.0,
                  spot_present=False, elongated_region_count=0, periodicity_strength=0.0,
                  stripe_present=False):
    return {
        "image_key": image_key, "is_reference": str(is_reference),
        "dominant_fraction": dominant_fraction, "hue_dispersion": hue_dispersion,
        "is_solid": str(is_solid), "n_significant_colors": n_significant_colors,
        "spot_count": spot_count, "mean_spot_area": mean_spot_area,
        "spot_present": str(spot_present),
        "elongated_region_count": elongated_region_count,
        "periodicity_strength": periodicity_strength, "stripe_present": str(stripe_present),
    }


def _config(tmp_path: Path, **overrides) -> AggregationConfig:
    defaults = {
        "pattern_features_csv_path": tmp_path / "pattern_features.csv",
        "species_coverage_csv_path": tmp_path / "species_coverage.csv",
        "extracted_root": tmp_path / "extracted_fish",
        "output_csv_path": tmp_path / "species_features.csv",
    }
    defaults.update(overrides)
    return AggregationConfig(**defaults)


def test_restricts_to_matched_species_only(tmp_path: Path):
    config = _config(tmp_path)
    _write_coverage_csv(config.species_coverage_csv_path, ["Acanthurus_achilles"])
    _write_features_csv(config.pattern_features_csv_path, [
        _feature_row("Acanthurus/Acanthurus_achilles/001"),
        _feature_row("Naso/Naso_unicornis/001"),  # not in the matched set
    ])
    _write_mask(config.extracted_root, "Acanthurus/Acanthurus_achilles/001", 0.5)

    aggregates = aggregate_species_features(config)

    assert [a.species for a in aggregates] == ["Acanthurus_achilles"]


def test_excludes_reference_image_by_default(tmp_path: Path):
    config = _config(tmp_path)
    _write_coverage_csv(config.species_coverage_csv_path, ["Acanthurus_achilles"])
    _write_features_csv(config.pattern_features_csv_path, [
        _feature_row("Acanthurus/Acanthurus_achilles/000_reference", is_reference=True),
        _feature_row("Acanthurus/Acanthurus_achilles/001"),
    ])
    _write_mask(config.extracted_root, "Acanthurus/Acanthurus_achilles/000_reference", 0.5)
    _write_mask(config.extracted_root, "Acanthurus/Acanthurus_achilles/001", 0.5)

    aggregates = aggregate_species_features(config)

    assert aggregates[0].n_images == 1


def test_include_reference_images_true_counts_it(tmp_path: Path):
    config = _config(tmp_path, include_reference_images=True)
    _write_coverage_csv(config.species_coverage_csv_path, ["Acanthurus_achilles"])
    _write_features_csv(config.pattern_features_csv_path, [
        _feature_row("Acanthurus/Acanthurus_achilles/000_reference", is_reference=True),
        _feature_row("Acanthurus/Acanthurus_achilles/001"),
    ])
    _write_mask(config.extracted_root, "Acanthurus/Acanthurus_achilles/000_reference", 0.5)
    _write_mask(config.extracted_root, "Acanthurus/Acanthurus_achilles/001", 0.5)

    aggregates = aggregate_species_features(config)

    assert aggregates[0].n_images == 2


def test_species_whose_only_image_is_the_reference_is_dropped(tmp_path: Path):
    # The real Naso maculatus case: excluding is_reference rows can drop a
    # species entirely, not just shrink its aggregate.
    config = _config(tmp_path)
    _write_coverage_csv(config.species_coverage_csv_path, ["Naso_maculatus"])
    _write_features_csv(config.pattern_features_csv_path, [
        _feature_row("Naso/Naso_maculatus/000_reference", is_reference=True),
    ])
    _write_mask(config.extracted_root, "Naso/Naso_maculatus/000_reference", 0.5)

    aggregates = aggregate_species_features(config)

    assert aggregates == []


def test_min_images_per_species_drops_sparse_species(tmp_path: Path):
    config = _config(tmp_path, min_images_per_species=3)
    _write_coverage_csv(config.species_coverage_csv_path, ["Acanthurus_achilles"])
    _write_features_csv(config.pattern_features_csv_path, [
        _feature_row("Acanthurus/Acanthurus_achilles/001"),
        _feature_row("Acanthurus/Acanthurus_achilles/002"),
    ])
    _write_mask(config.extracted_root, "Acanthurus/Acanthurus_achilles/001", 0.5)
    _write_mask(config.extracted_root, "Acanthurus/Acanthurus_achilles/002", 0.5)

    aggregates = aggregate_species_features(config)

    assert aggregates == []


def test_boolean_features_aggregate_as_proportion(tmp_path: Path):
    config = _config(tmp_path)
    _write_coverage_csv(config.species_coverage_csv_path, ["Acanthurus_achilles"])
    _write_features_csv(config.pattern_features_csv_path, [
        _feature_row("Acanthurus/Acanthurus_achilles/001", stripe_present=True),
        _feature_row("Acanthurus/Acanthurus_achilles/002", stripe_present=True),
        _feature_row("Acanthurus/Acanthurus_achilles/003", stripe_present=False),
        _feature_row("Acanthurus/Acanthurus_achilles/004", stripe_present=False),
    ])
    for i in range(1, 5):
        _write_mask(config.extracted_root, f"Acanthurus/Acanthurus_achilles/00{i}", 0.5)

    aggregates = aggregate_species_features(config)

    assert aggregates[0].prop_striped == 0.5


def test_numeric_features_aggregate_as_arithmetic_mean(tmp_path: Path):
    config = _config(tmp_path)
    _write_coverage_csv(config.species_coverage_csv_path, ["Acanthurus_achilles"])
    _write_features_csv(config.pattern_features_csv_path, [
        _feature_row("Acanthurus/Acanthurus_achilles/001", dominant_fraction=0.2),
        _feature_row("Acanthurus/Acanthurus_achilles/002", dominant_fraction=0.8),
    ])
    _write_mask(config.extracted_root, "Acanthurus/Acanthurus_achilles/001", 0.5)
    _write_mask(config.extracted_root, "Acanthurus/Acanthurus_achilles/002", 0.5)

    aggregates = aggregate_species_features(config)

    assert aggregates[0].mean_dominant_fraction == pytest.approx(0.5)


def test_spot_area_normalized_by_each_images_own_masked_pixel_count(tmp_path: Path):
    config = _config(tmp_path)
    _write_coverage_csv(config.species_coverage_csv_path, ["Acanthurus_achilles"])
    # Image 1: 100 total px, 50% masked (50 masked px), mean_spot_area=5 -> fraction 0.1
    # Image 2: 100 total px, 25% masked (25 masked px), mean_spot_area=5 -> fraction 0.2
    _write_features_csv(config.pattern_features_csv_path, [
        _feature_row("Acanthurus/Acanthurus_achilles/001", mean_spot_area=5.0),
        _feature_row("Acanthurus/Acanthurus_achilles/002", mean_spot_area=5.0),
    ])
    _write_mask(config.extracted_root, "Acanthurus/Acanthurus_achilles/001", 0.5)
    _write_mask(config.extracted_root, "Acanthurus/Acanthurus_achilles/002", 0.25)

    aggregates = aggregate_species_features(config)

    assert aggregates[0].mean_spot_area_fraction == pytest.approx((0.1 + 0.2) / 2)


def test_write_species_features_csv_round_trips(tmp_path: Path):
    from distance_matrices.aggregation import write_species_features_csv

    config = _config(tmp_path)
    _write_coverage_csv(config.species_coverage_csv_path, ["Acanthurus_achilles"])
    _write_features_csv(config.pattern_features_csv_path, [
        _feature_row("Acanthurus/Acanthurus_achilles/001"),
    ])
    _write_mask(config.extracted_root, "Acanthurus/Acanthurus_achilles/001", 0.5)

    aggregates = aggregate_species_features(config)
    write_species_features_csv(aggregates, config.output_csv_path)

    with open(config.output_csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["species"] == "Acanthurus_achilles"
    assert rows[0]["n_images"] == "1"
