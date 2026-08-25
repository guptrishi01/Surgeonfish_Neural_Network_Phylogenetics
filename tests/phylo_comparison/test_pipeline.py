"""Integration tests for the full Phase 4 preparation pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

from distance_matrices.aggregation import SpeciesAggregate, write_species_features_csv
from phylo_comparison.config import ExportConfig, FeaturePrepConfig
from phylo_comparison.feature_prep import DIMENSIONS
from phylo_comparison.pipeline import run


def _aggregate(species, n_images=10, **overrides):
    defaults = {
        "species": species, "n_images": n_images,
        "mean_dominant_fraction": 0.5, "mean_hue_dispersion": 0.1,
        "mean_n_significant_colors": 2, "prop_solid": 0.5,
        "mean_elongated_region_count": 5, "mean_periodicity_strength": 0.2, "prop_striped": 0.3,
        "mean_spot_count": 1, "mean_spot_area_fraction": 0.01, "prop_spotted": 0.1,
    }
    defaults.update(overrides)
    return SpeciesAggregate(**defaults)


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


def test_full_prep_pipeline_writes_every_matrix_and_the_tree(tmp_path: Path):
    # 6 species, more than the widest dimension's 4 features, so every
    # dimension's matrix stays well-conditioned after centering (a matrix
    # with rows <= columns is rank-deficient by construction) - every
    # numeric feature varies linearly with i so no column ends up constant
    # (a zero-variance column is equally rank-deficient).
    species = [
        "Acanthurus_achilles", "Ctenochaetus_striatus", "Naso_unicornis",
        "Prionurus_laticlavius", "Zebrasoma_scopas", "Zebrasoma_veliferum",
    ]
    aggregates = [
        _aggregate(
            name,
            mean_dominant_fraction=0.2 + 0.1 * i, mean_hue_dispersion=0.05 + 0.05 * i,
            mean_n_significant_colors=1 + i, prop_solid=0.1 + 0.15 * i,
            mean_elongated_region_count=1 + 5 * i, mean_periodicity_strength=0.05 + 0.05 * i,
            prop_striped=0.05 * i,
            mean_spot_count=1 + i, mean_spot_area_fraction=0.005 + 0.002 * i,
            prop_spotted=0.1 + 0.1 * i,
        )
        for i, name in enumerate(species)
    ]
    features_csv = tmp_path / "reports" / "species_features.csv"
    write_species_features_csv(aggregates, features_csv)

    coverage_csv = tmp_path / "species_coverage.csv"
    _write_coverage_csv(coverage_csv, species)

    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text(
        "(((Acanthurus_achilles:1,Ctenochaetus_striatus:1):1,"
        "(Naso_unicornis:1,Prionurus_laticlavius:1):1):1,"
        "(Zebrasoma_scopas:1,Zebrasoma_veliferum:1):1);",
        encoding="utf-8",
    )

    feature_prep_config = FeaturePrepConfig(species_features_csv_path=features_csv)
    export_config = ExportConfig(
        output_dir=tmp_path / "outputs" / "phase4",
        tree_path=tree_path,
        species_coverage_csv_path=coverage_csv,
        min_tip_count=2,
        max_tip_count=10,
    )

    species_order = run(feature_prep_config, export_config)

    assert species_order == sorted(species)
    for dimension in DIMENSIONS:
        assert (export_config.output_dir / f"{dimension}_kmult_features.csv").exists()
    assert (export_config.output_dir / "pruned_tree.nwk").exists()
