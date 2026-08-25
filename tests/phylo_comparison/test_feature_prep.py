"""Unit tests for Kmult feature-matrix admissibility transforms."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from distance_matrices.aggregation import SpeciesAggregate
from phylo_comparison.config import FeaturePrepConfig
from phylo_comparison.feature_prep import (
    load_species_aggregates,
    logit,
    prepare_dimension_matrix,
    smithson_verkuilen_adjust,
)


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


def test_smithson_verkuilen_shrinks_toward_half_scaled_by_trial_count():
    proportions = np.array([0.0, 1.0, 0.5])
    n_trials = np.array([10.0, 10.0, 10.0])

    adjusted = smithson_verkuilen_adjust(proportions, n_trials)

    assert adjusted == pytest.approx([0.05, 0.95, 0.5])


def test_smithson_verkuilen_with_one_trial_always_gives_half():
    adjusted = smithson_verkuilen_adjust(np.array([0.0, 1.0]), np.array([1.0, 1.0]))

    assert adjusted == pytest.approx([0.5, 0.5])


def test_smithson_verkuilen_shrinks_less_with_more_trials():
    small_n = smithson_verkuilen_adjust(np.array([0.0]), np.array([2.0]))
    large_n = smithson_verkuilen_adjust(np.array([0.0]), np.array([100.0]))

    assert small_n[0] > large_n[0] > 0.0


def test_logit_is_antisymmetric_around_half():
    p = np.array([0.05, 0.5, 0.95])

    result = logit(p)

    assert result[1] == pytest.approx(0.0)
    assert result[0] == pytest.approx(-result[2])


def test_prepare_dimension_matrix_handles_a_genuine_zero_proportion_without_crashing():
    # Several real species have prop_solid/prop_striped/prop_spotted == 0.0
    # exactly - a raw logit(0) would be -inf without the SV adjustment.
    # Needs more species than features (3) to stay well-conditioned after
    # centering - a matrix with rows <= columns is rank-deficient by
    # construction regardless of data quality, see the dedicated
    # ill-conditioned test below.
    stripe_props = [0.0, 1.0, 0.5, 0.2, 0.8]  # includes a genuine 0.0 and 1.0
    aggregates = [
        _aggregate(
            chr(ord("A") + i), prop_striped=p,
            mean_elongated_region_count=1 + 6 * i, mean_periodicity_strength=0.1 + 0.05 * i,
        )
        for i, p in enumerate(stripe_props)
    ]
    config = FeaturePrepConfig()

    prepared = prepare_dimension_matrix(aggregates, "stripe", config)

    assert np.all(np.isfinite(prepared.matrix))


def test_prepare_dimension_matrix_rejects_unknown_dimension():
    with pytest.raises(ValueError):
        prepare_dimension_matrix([_aggregate("A")], "nonsense", FeaturePrepConfig())


def test_prepare_dimension_matrix_sets_pac_no_to_full_feature_count():
    # Needs more species than features (4, for "color") to stay
    # well-conditioned after centering.
    aggregates = [
        _aggregate("A", mean_dominant_fraction=0.2, mean_hue_dispersion=0.05,
                   mean_n_significant_colors=1, prop_solid=0.9),
        _aggregate("B", mean_dominant_fraction=0.4, mean_hue_dispersion=0.15,
                   mean_n_significant_colors=2, prop_solid=0.6),
        _aggregate("C", mean_dominant_fraction=0.6, mean_hue_dispersion=0.25,
                   mean_n_significant_colors=3, prop_solid=0.3),
        _aggregate("D", mean_dominant_fraction=0.8, mean_hue_dispersion=0.35,
                   mean_n_significant_colors=4, prop_solid=0.1),
        _aggregate("E", mean_dominant_fraction=0.5, mean_hue_dispersion=0.2,
                   mean_n_significant_colors=2, prop_solid=0.5),
        _aggregate("F", mean_dominant_fraction=0.3, mean_hue_dispersion=0.1,
                   mean_n_significant_colors=1, prop_solid=0.7),
    ]
    config = FeaturePrepConfig()

    prepared = prepare_dimension_matrix(aggregates, "color", config)

    assert prepared.pac_no == len(prepared.feature_names) == 4


def test_prepare_dimension_matrix_raises_on_ill_conditioned_input():
    # Two species with numerically-identical feature vectors across every
    # column produce a rank-deficient (singular) matrix.
    aggregates = [
        _aggregate("A", mean_dominant_fraction=0.5, mean_hue_dispersion=0.5,
                   mean_n_significant_colors=2, prop_solid=0.5),
        _aggregate("B", mean_dominant_fraction=0.5, mean_hue_dispersion=0.5,
                   mean_n_significant_colors=2, prop_solid=0.5),
        _aggregate("C", mean_dominant_fraction=0.5, mean_hue_dispersion=0.5,
                   mean_n_significant_colors=2, prop_solid=0.5),
    ]
    config = FeaturePrepConfig(condition_number_threshold=1e6)

    with pytest.raises(ValueError):
        prepare_dimension_matrix(aggregates, "color", config)


def test_prepare_dimension_matrix_standardizes_to_zero_mean():
    # Needs more species than features (3, for "stripe") to stay
    # well-conditioned after centering.
    aggregates = [
        _aggregate(
            chr(ord("A") + i),
            mean_elongated_region_count=1 + 6 * i, mean_periodicity_strength=0.05 + 0.1 * i,
            prop_striped=0.1 + 0.2 * i,
        )
        for i in range(4)
    ]
    config = FeaturePrepConfig()

    prepared = prepare_dimension_matrix(aggregates, "stripe", config)

    assert np.allclose(prepared.matrix.mean(axis=0), 0.0, atol=1e-9)


def test_load_species_aggregates_round_trips_from_csv(tmp_path: Path):
    from distance_matrices.aggregation import write_species_features_csv

    aggregates = [_aggregate("Acanthurus_achilles", n_images=21)]
    csv_path = tmp_path / "species_features.csv"
    write_species_features_csv(aggregates, csv_path)

    loaded = load_species_aggregates(csv_path)

    assert loaded[0].species == "Acanthurus_achilles"
    assert loaded[0].n_images == 21


def test_load_species_aggregates_handles_multiple_rows(tmp_path: Path):
    import io

    csv_path = tmp_path / "species_features.csv"
    fieldnames = [
        "species", "n_images",
        "mean_dominant_fraction", "mean_hue_dispersion", "mean_n_significant_colors", "prop_solid",
        "mean_elongated_region_count", "mean_periodicity_strength", "prop_striped",
        "mean_spot_count", "mean_spot_area_fraction", "prop_spotted",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for species in ["A", "B"]:
        writer.writerow({
            "species": species, "n_images": 5,
            "mean_dominant_fraction": 0.5, "mean_hue_dispersion": 0.1,
            "mean_n_significant_colors": 2, "prop_solid": 0.5,
            "mean_elongated_region_count": 5, "mean_periodicity_strength": 0.2, "prop_striped": 0.3,
            "mean_spot_count": 1, "mean_spot_area_fraction": 0.01, "prop_spotted": 0.1,
        })
    csv_path.write_text(buffer.getvalue(), encoding="utf-8")

    loaded = load_species_aggregates(csv_path)

    assert [a.species for a in loaded] == ["A", "B"]
