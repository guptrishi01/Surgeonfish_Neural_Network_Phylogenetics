"""Integration tests for the full Phase 3 pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from distance_matrices.config import AggregationConfig, DistanceMatrixConfig, PhylogenyConfig
from distance_matrices.pipeline import DIMENSIONS, run

_FEATURE_FIELDNAMES = [
    "image_key", "is_reference", "dominant_fraction", "hue_dispersion", "is_solid",
    "n_significant_colors", "spot_count", "mean_spot_area", "spot_present",
    "elongated_region_count", "periodicity_strength", "stripe_present",
]


def _write_features_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FEATURE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _feature_row(image_key: str, **overrides) -> dict:
    row = {
        "image_key": image_key, "is_reference": "False", "dominant_fraction": 0.5,
        "hue_dispersion": 0.1, "is_solid": "True", "n_significant_colors": 1,
        "spot_count": 0, "mean_spot_area": 0.0, "spot_present": "False",
        "elongated_region_count": 0, "periodicity_strength": 0.0, "stripe_present": "False",
    }
    row.update(overrides)
    return row


def _write_coverage_csv(path: Path, species_list: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["species", "genus", "has_genetic_data", "match_method"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sp in species_list:
            writer.writerow({
                "species": sp.replace("_", " "), "genus": sp.split("_")[0],
                "has_genetic_data": "yes", "match_method": "exact",
            })


def _write_mask(extracted_root: Path, image_key: str, size=(10, 10)) -> None:
    path = extracted_root / f"{image_key}_mask.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.full(size, 255, dtype=np.uint8)
    Image.fromarray(mask).save(path)


def test_full_pipeline_writes_aligned_matrices_for_every_species(tmp_path: Path):
    species = ["Acanthurus_achilles", "Naso_unicornis", "Zebrasoma_scopas"]
    _write_coverage_csv(tmp_path / "species_coverage.csv", species)

    feature_rows = []
    for sp in species:
        genus = sp.split("_")[0]
        for i in range(1, 4):
            key = f"{genus}/{sp}/00{i}"
            feature_rows.append(_feature_row(key, dominant_fraction=0.1 * i))
            _write_mask(tmp_path / "extracted_fish", key)
    _write_features_csv(tmp_path / "pattern_features.csv", feature_rows)

    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text(
        "((Acanthurus_achilles:1,Naso_unicornis:1):1,Zebrasoma_scopas:2);", encoding="utf-8"
    )

    aggregation_config = AggregationConfig(
        pattern_features_csv_path=tmp_path / "pattern_features.csv",
        species_coverage_csv_path=tmp_path / "species_coverage.csv",
        extracted_root=tmp_path / "extracted_fish",
        output_csv_path=tmp_path / "reports" / "species_features.csv",
    )
    distance_config = DistanceMatrixConfig(output_dir=tmp_path / "outputs")
    phylogeny_config = PhylogenyConfig(
        tree_path=tree_path,
        species_coverage_csv_path=tmp_path / "species_coverage.csv",
        min_tip_count=2,
        max_tip_count=10,
        output_path=tmp_path / "outputs" / "patristic_distance_matrix.csv",
    )

    result = run(aggregation_config, distance_config, phylogeny_config)

    assert result.species_order == sorted(species)
    assert len(result.aggregates) == 3
    for dimension in DIMENSIONS:
        assert result.pattern_distance_matrices[dimension].shape == (3, 3)
        assert (distance_config.output_dir / f"{dimension}_distance_matrix.csv").exists()
    assert result.patristic_distance_matrix.shape == (3, 3)
    assert phylogeny_config.output_path.exists()
    assert (tmp_path / "reports" / "species_features.csv").exists()


def test_pipeline_handles_a_species_with_only_a_reference_image(tmp_path: Path):
    # The real Naso maculatus scenario end to end: default
    # include_reference_images=False drops it from every output.
    species = ["Acanthurus_achilles", "Naso_maculatus"]
    _write_coverage_csv(tmp_path / "species_coverage.csv", species)

    feature_rows = [
        _feature_row("Acanthurus/Acanthurus_achilles/001"),
        _feature_row("Naso/Naso_maculatus/000_reference", is_reference="True"),
    ]
    _write_mask(tmp_path / "extracted_fish", "Acanthurus/Acanthurus_achilles/001")
    _write_mask(tmp_path / "extracted_fish", "Naso/Naso_maculatus/000_reference")
    _write_features_csv(tmp_path / "pattern_features.csv", feature_rows)

    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text("(Acanthurus_achilles:1,Naso_maculatus:1);", encoding="utf-8")

    aggregation_config = AggregationConfig(
        pattern_features_csv_path=tmp_path / "pattern_features.csv",
        species_coverage_csv_path=tmp_path / "species_coverage.csv",
        extracted_root=tmp_path / "extracted_fish",
        output_csv_path=tmp_path / "reports" / "species_features.csv",
    )
    distance_config = DistanceMatrixConfig(output_dir=tmp_path / "outputs")
    phylogeny_config = PhylogenyConfig(
        tree_path=tree_path,
        species_coverage_csv_path=tmp_path / "species_coverage.csv",
        min_tip_count=1,
        max_tip_count=10,
        output_path=tmp_path / "outputs" / "patristic_distance_matrix.csv",
    )

    result = run(aggregation_config, distance_config, phylogeny_config)

    assert result.species_order == ["Acanthurus_achilles"]
    assert result.patristic_distance_matrix.shape == (1, 1)
