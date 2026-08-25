"""Coloring-dimension features: cluster fractions plus a direct, unpartitioned
hue/saturation dispersion measure for solid-colour classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pattern_extractor.clustering import ClusterResult
from pattern_extractor.color_space import rgb_to_hue_sat_vector
from pattern_extractor.config import ColorConfig


@dataclass
class ColorFeatures:
    """Coloring-dimension features for one image.

    Attributes:
        palette_rgb: (k, 3) array of cluster-centre colours.
        fractions: (k,) array of each colour's share of the fish body.
        dominant_fraction: Largest single-colour fraction (informational -
            not used for is_solid, see hue_dispersion).
        hue_dispersion: Mean distance of every masked-in pixel's
            hue/saturation vector from their collective mean - low for a
            genuinely solid-coloured fish, high for a multi-coloured one.
            See ColorConfig.max_hue_dispersion_for_solid for why this
            replaced a cluster-dominance check.
        is_solid: True if hue_dispersion <= config threshold.
        n_significant_colors: Count of clusters covering a non-trivial
            share of the body (a simple "how many colours" signal).
    """

    palette_rgb: np.ndarray
    fractions: np.ndarray
    dominant_fraction: float
    hue_dispersion: float
    is_solid: bool
    n_significant_colors: int


def extract_color_features(
    image_rgb: np.ndarray, mask: np.ndarray, cluster_result: ClusterResult, config: ColorConfig
) -> ColorFeatures:
    """Summarizes a masked image and its ClusterResult into coloring-dimension features.

    Args:
        image_rgb: (H, W, 3) uint8 RGB array.
        mask: (H, W) boolean array, True for fish (masked-in) pixels.
        cluster_result: Output of clustering.assign_clusters().
        config: Coloring thresholds.

    Returns:
        ColorFeatures for this image.
    """
    fractions = cluster_result.fractions
    dominant_fraction = float(fractions.max()) if fractions.size else 0.0
    n_significant = int(np.sum(fractions >= 0.05))

    pixels_rgb = image_rgb[mask].astype(float)
    if len(pixels_rgb) > 0:
        vectors = rgb_to_hue_sat_vector(pixels_rgb)
        hue_dispersion = float(np.linalg.norm(vectors - vectors.mean(axis=0), axis=1).mean())
    else:
        hue_dispersion = 0.0
    is_solid = hue_dispersion <= config.max_hue_dispersion_for_solid

    return ColorFeatures(
        palette_rgb=cluster_result.centers,
        fractions=fractions,
        dominant_fraction=dominant_fraction,
        hue_dispersion=hue_dispersion,
        is_solid=is_solid,
        n_significant_colors=n_significant,
    )
