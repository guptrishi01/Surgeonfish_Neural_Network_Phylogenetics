"""Feature-matrix admissibility transforms for the Kmult phylogenetic-signal test.

See the package docstring for why these specific transforms were chosen -
particularly why no compositional (CLR/ILR) transform is applied (no
feature in the actual exported set is a true composition) and why two
different [0, 1]-bounded feature types get two different corrections.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from distance_matrices.aggregation import SpeciesAggregate
from phylo_comparison.config import FeaturePrepConfig

logger = logging.getLogger(__name__)

DIMENSIONS = ["color", "stripe", "spot"]

# Mirrors distance_matrices.distance's dimension groupings (same
# underlying per-species columns) - kept as a separate mapping since this
# module's transforms serve a different statistical purpose (Kmult
# admissibility, not Euclidean-distance preparation) and the two phases
# are deliberately decoupled packages.
_DIMENSION_FEATURES = {
    "color": [
        "mean_dominant_fraction", "mean_hue_dispersion", "mean_n_significant_colors", "prop_solid",
    ],
    "stripe": ["mean_elongated_region_count", "mean_periodicity_strength", "prop_striped"],
    "spot": ["mean_spot_count", "mean_spot_area_fraction", "prop_spotted"],
}

# True proportions-of-count (a fraction of a species' own images scoring
# True) - get the Smithson & Verkuilen boundary adjustment before a logit
# transform. Every other feature (means of continuous per-image values,
# or counts/magnitudes) gets log1p instead - see the package docstring.
_PROPORTION_OF_COUNT_FEATURES = {"prop_solid", "prop_striped", "prop_spotted"}


def smithson_verkuilen_adjust(proportions: np.ndarray, n_trials: np.ndarray) -> np.ndarray:
    """Shrinks proportions away from the 0/1 boundary, scaled by each observation's trial count.

    Smithson, M., & Verkuilen, J. (2006). A better lemon squeezer?
    Maximum-likelihood regression with beta-distributed dependent
    variables. Psychological Methods, 11(1), 54-71. Designed specifically
    for proportion-of-count data with genuine 0/1 observations - unlike a
    fixed small epsilon, the amount of shrinkage naturally scales down as
    the trial count grows, since a proportion estimated from more trials
    is more precise and needs less correction. Real motivation here, not
    a generic safety clip: several species genuinely have
    prop_solid/prop_striped/prop_spotted == 0.0 (checked directly against
    reports/species_features.csv), which a raw logit(0) would send to
    -inf.

    Args:
        proportions: Array of proportions in [0, 1].
        n_trials: Array of trial counts (each species' n_images) the
            corresponding proportion was computed from, same shape.

    Returns:
        Adjusted proportions in the open interval (0, 1).
    """
    return (proportions * (n_trials - 1.0) + 0.5) / n_trials


def logit(p: np.ndarray) -> np.ndarray:
    """Log-odds transform, mapping (0, 1) to the real line.

    Args:
        p: Array of values strictly inside (0, 1) - see
            ``smithson_verkuilen_adjust()`` for guaranteeing this for
            proportion-of-count features before calling this.

    Returns:
        log(p / (1 - p)), same shape.
    """
    return np.log(p / (1.0 - p))


@dataclass
class PreparedMatrix:
    """A Kmult-ready feature matrix for one pattern dimension.

    Attributes:
        dimension: "color", "stripe", or "spot".
        species: Row labels, in matrix order.
        feature_names: Column labels (the original feature names, before
            transform), in matrix order.
        matrix: (n_species, n_features) transformed and standardized
            array.
        pac_no: Number of components to request from physignal.z - set to
            n_features (no PCA reduction), since every actual dimension's
            feature count is already far below n_species. See the package
            docstring for why this deviates from the shape-data-many-
            landmarks default the geomorph manual assumes.
        condition_number: The matrix's condition number
            (``numpy.linalg.cond``), recorded for the R side to report
            alongside results, not just checked once here.
    """

    dimension: str
    species: list[str]
    feature_names: list[str]
    matrix: np.ndarray
    pac_no: int
    condition_number: float


def load_species_aggregates(species_features_csv_path: Path) -> list[SpeciesAggregate]:
    """Loads already-aggregated per-species features from Phase 3's output CSV.

    Args:
        species_features_csv_path: Phase 3's output
            (``distance_matrices.aggregation.write_species_features_csv()``).

    Returns:
        One SpeciesAggregate per row, in file order (Phase 3 already
        writes them sorted by species).
    """
    with open(species_features_csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        SpeciesAggregate(
            species=row["species"],
            n_images=int(row["n_images"]),
            mean_dominant_fraction=float(row["mean_dominant_fraction"]),
            mean_hue_dispersion=float(row["mean_hue_dispersion"]),
            mean_n_significant_colors=float(row["mean_n_significant_colors"]),
            prop_solid=float(row["prop_solid"]),
            mean_elongated_region_count=float(row["mean_elongated_region_count"]),
            mean_periodicity_strength=float(row["mean_periodicity_strength"]),
            prop_striped=float(row["prop_striped"]),
            mean_spot_count=float(row["mean_spot_count"]),
            mean_spot_area_fraction=float(row["mean_spot_area_fraction"]),
            prop_spotted=float(row["prop_spotted"]),
        )
        for row in rows
    ]


def prepare_dimension_matrix(
    aggregates: list[SpeciesAggregate], dimension: str, config: FeaturePrepConfig
) -> PreparedMatrix:
    """Builds a Kmult-admissible feature matrix for one pattern dimension.

    Args:
        aggregates: Output of ``load_species_aggregates()`` (or
            ``distance_matrices.aggregation.aggregate_species_features()``
            directly).
        dimension: One of "color", "stripe", "spot".
        config: Preparation settings.

    Returns:
        A PreparedMatrix.

    Raises:
        ValueError: If dimension isn't recognized, or if the resulting
            matrix is ill-conditioned (condition number at or above
            ``config.condition_number_threshold``) - failing loudly
            rather than handing R a numerically unstable input, per the
            README's Planned Approach step 4 requirement.
    """
    if dimension not in _DIMENSION_FEATURES:
        raise ValueError(
            f"Unknown dimension {dimension!r}, expected one of {list(_DIMENSION_FEATURES)}"
        )
    feature_names = _DIMENSION_FEATURES[dimension]
    species = [a.species for a in aggregates]
    n_images = np.array([a.n_images for a in aggregates], dtype=float)

    columns = []
    for name in feature_names:
        raw = np.array([getattr(a, name) for a in aggregates], dtype=float)
        if name in _PROPORTION_OF_COUNT_FEATURES:
            adjusted = smithson_verkuilen_adjust(raw, n_images)
            columns.append(logit(adjusted))
        else:
            columns.append(np.log1p(raw))
    transformed = np.column_stack(columns)

    mean = transformed.mean(axis=0)
    std = transformed.std(axis=0)
    safe_std = np.where(std > 1e-12, std, 1.0)
    standardized = (transformed - mean) / safe_std

    condition_number = float(np.linalg.cond(standardized))
    if condition_number >= config.condition_number_threshold:
        raise ValueError(
            f"{dimension} feature matrix is ill-conditioned (condition number "
            f"{condition_number:.3e} >= threshold {config.condition_number_threshold:.3e}) - "
            "refusing to hand this to physignal.z rather than risk a silently unstable result."
        )

    return PreparedMatrix(
        dimension=dimension,
        species=species,
        feature_names=feature_names,
        matrix=standardized,
        pac_no=len(feature_names),
        condition_number=condition_number,
    )
