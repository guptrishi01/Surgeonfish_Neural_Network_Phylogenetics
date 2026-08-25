"""Unit tests for distance matrix construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from distance_matrices.aggregation import SpeciesAggregate
from distance_matrices.distance import (
    build_feature_matrix,
    pairwise_distance_matrix,
    standardize,
    write_distance_matrix_csv,
)


def _aggregate(species, **overrides):
    defaults = {
        "species": species, "n_images": 10,
        "mean_dominant_fraction": 0.5, "mean_hue_dispersion": 0.1,
        "mean_n_significant_colors": 2, "prop_solid": 0.5,
        "mean_elongated_region_count": 5, "mean_periodicity_strength": 0.2, "prop_striped": 0.3,
        "mean_spot_count": 1, "mean_spot_area_fraction": 0.01, "prop_spotted": 0.1,
    }
    defaults.update(overrides)
    return SpeciesAggregate(**defaults)


def test_build_feature_matrix_extracts_the_right_columns_per_dimension():
    aggregates = [
        _aggregate("A", mean_dominant_fraction=0.1),
        _aggregate("B", mean_dominant_fraction=0.9),
    ]

    species, matrix = build_feature_matrix(aggregates, "color")

    assert species == ["A", "B"]
    assert matrix.shape == (2, 4)
    assert matrix[0, 0] == 0.1
    assert matrix[1, 0] == 0.9


def test_build_feature_matrix_rejects_unknown_dimension():
    with pytest.raises(ValueError):
        build_feature_matrix([_aggregate("A")], "nonsense")


def test_standardize_gives_zero_mean_unit_variance_columns():
    matrix = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    result = standardize(matrix)

    assert np.allclose(result.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(result.std(axis=0), 1.0, atol=1e-9)


def test_standardize_handles_zero_variance_column_without_dividing_by_zero():
    matrix = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])

    result = standardize(matrix)

    assert np.allclose(result[:, 0], 0.0)


def test_pairwise_distance_matrix_is_symmetric_with_zero_diagonal():
    matrix = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    distances = pairwise_distance_matrix(matrix)

    assert distances.shape == (3, 3)
    assert np.allclose(np.diag(distances), 0.0)
    assert np.allclose(distances, distances.T)
    assert distances[0, 1] == pytest.approx(1.0)


def test_write_distance_matrix_csv_round_trips(tmp_path: Path):
    import csv

    species = ["A", "B"]
    matrix = np.array([[0.0, 1.5], [1.5, 0.0]])
    path = tmp_path / "matrix.csv"

    write_distance_matrix_csv(species, matrix, path)

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["", "A", "B"]
    assert rows[1][0] == "A"
    assert float(rows[1][2]) == pytest.approx(1.5)
