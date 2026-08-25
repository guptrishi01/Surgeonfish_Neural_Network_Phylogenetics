"""Pairwise species x species distance matrices from per-species feature vectors.

Euclidean distance on z-score-standardized features - see the package
docstring for why this metric was chosen over Bray-Curtis.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

from distance_matrices.aggregation import SpeciesAggregate

logger = logging.getLogger(__name__)

_DIMENSION_FEATURES = {
    "color": [
        "mean_dominant_fraction", "mean_hue_dispersion", "mean_n_significant_colors", "prop_solid",
    ],
    "stripe": ["mean_elongated_region_count", "mean_periodicity_strength", "prop_striped"],
    "spot": ["mean_spot_count", "mean_spot_area_fraction", "prop_spotted"],
}


def build_feature_matrix(
    aggregates: list[SpeciesAggregate], dimension: str
) -> tuple[list[str], np.ndarray]:
    """Extracts one pattern dimension's feature columns into a species x feature matrix.

    Args:
        aggregates: Output of ``aggregation.aggregate_species_features()``.
        dimension: One of "color", "stripe", "spot".

    Returns:
        (species list, (n_species, n_features) float array), in the same
        species order as `aggregates`.

    Raises:
        ValueError: If dimension isn't recognized.
    """
    if dimension not in _DIMENSION_FEATURES:
        raise ValueError(
            f"Unknown dimension {dimension!r}, expected one of {list(_DIMENSION_FEATURES)}"
        )
    feature_names = _DIMENSION_FEATURES[dimension]
    species = [a.species for a in aggregates]
    matrix = np.array(
        [[getattr(a, name) for name in feature_names] for a in aggregates], dtype=float
    )
    return species, matrix


def standardize(matrix: np.ndarray) -> np.ndarray:
    """Z-score standardizes each column (feature) independently.

    A zero-variance column (e.g. every species scored identically on some
    feature) is left at zero after centering rather than divided by zero.

    Args:
        matrix: (n_species, n_features) array.

    Returns:
        Standardized array, same shape.
    """
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    safe_std = np.where(std > 1e-12, std, 1.0)
    return (matrix - mean) / safe_std


def pairwise_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance matrix on (typically already-standardized) rows.

    Args:
        matrix: (n_species, n_features) array.

    Returns:
        (n_species, n_species) symmetric distance matrix, zero diagonal.
    """
    return squareform(pdist(matrix, metric="euclidean"))


def write_distance_matrix_csv(species: list[str], matrix: np.ndarray, output_path: Path) -> None:
    """Writes a species x species distance matrix as a labeled CSV.

    Args:
        species: Row/column labels, in matrix order.
        matrix: (n_species, n_species) array.
        output_path: Where to write the CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + species)
        for name, row in zip(species, matrix):
            writer.writerow([name] + [f"{v:.6f}" for v in row])
    logger.info("Wrote %dx%d distance matrix to %s", len(species), len(species), output_path)
