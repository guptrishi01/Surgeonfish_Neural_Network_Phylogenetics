"""Coloring-dimension features derived from cluster fractions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pattern_extractor.clustering import ClusterResult
from pattern_extractor.config import ColorConfig


@dataclass
class ColorFeatures:
    """Coloring-dimension features for one image.

    Attributes:
        palette_rgb: (k, 3) array of cluster-centre colours.
        fractions: (k,) array of each colour's share of the fish body.
        dominant_fraction: Largest single-colour fraction.
        is_solid: True if dominant_fraction >= config threshold.
        n_significant_colors: Count of clusters covering a non-trivial
            share of the body (a simple "how many colours" signal).
    """

    palette_rgb: np.ndarray
    fractions: np.ndarray
    dominant_fraction: float
    is_solid: bool
    n_significant_colors: int


def extract_color_features(cluster_result: ClusterResult, config: ColorConfig) -> ColorFeatures:
    """Summarizes a ClusterResult into coloring-dimension features.

    Args:
        cluster_result: Output of clustering.assign_clusters().
        config: Coloring thresholds.

    Returns:
        ColorFeatures for this image.
    """
    fractions = cluster_result.fractions
    dominant_fraction = float(fractions.max()) if fractions.size else 0.0
    is_solid = dominant_fraction >= config.solid_dominant_fraction_threshold
    n_significant = int(np.sum(fractions >= 0.05))
    return ColorFeatures(
        palette_rgb=cluster_result.centers,
        fractions=fractions,
        dominant_fraction=dominant_fraction,
        is_solid=is_solid,
        n_significant_colors=n_significant,
    )
