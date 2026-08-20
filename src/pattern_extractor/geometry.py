"""Shared connected-component / region-shape analysis.

Used by both spot.py (roughly-round regions) and stripe.py (elongated
regions) to characterize a cluster's binary mask - this project's own
extension on top of patternize's colour-region output (see __init__.py).
No scikit-image dependency: labeling via scipy.ndimage.label, shape stats
(area, centroid, eccentricity, orientation) computed directly from image
moments rather than skimage.measure.regionprops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pattern_extractor.clustering import ClusterResult


@dataclass
class RegionStats:
    """Shape statistics for one connected component of a binary mask.

    Attributes:
        area: Pixel count.
        centroid: (row, col) of the region's center of mass.
        eccentricity: 0 for a circle, approaching 1 for a very elongated
            (line-like) shape.
        orientation_rad: Angle of the region's major axis, in radians,
            within [-pi/2, pi/2].
    """

    area: int
    centroid: tuple[float, float]
    eccentricity: float
    orientation_rad: float


def find_regions(binary_mask: np.ndarray, min_area: int = 1) -> list[RegionStats]:
    """Labels connected components and computes their shape statistics.

    Args:
        binary_mask: 2D boolean array.
        min_area: Components smaller than this (pixel count) are dropped as
            noise.

    Returns:
        One RegionStats per surviving connected component.
    """
    labeled, n = ndimage.label(binary_mask)
    regions = []
    for label_id in range(1, n + 1):
        rows, cols = np.where(labeled == label_id)
        area = len(rows)
        if area < min_area:
            continue
        row_mean, col_mean = rows.mean(), cols.mean()
        dr, dc = rows - row_mean, cols - col_mean
        mu20 = float(np.mean(dr * dr))
        mu02 = float(np.mean(dc * dc))
        mu11 = float(np.mean(dr * dc))
        common = math.sqrt((mu20 - mu02) ** 2 + 4 * mu11**2)
        lambda1 = (mu20 + mu02 + common) / 2
        lambda2 = (mu20 + mu02 - common) / 2
        ratio = lambda2 / lambda1 if lambda1 > 1e-9 else 0.0
        eccentricity = min(math.sqrt(max(0.0, 1.0 - ratio)), 1.0)
        orientation = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
        regions.append(
            RegionStats(
                area=area,
                centroid=(row_mean, col_mean),
                eccentricity=eccentricity,
                orientation_rad=orientation,
            )
        )
    return regions


def non_dominant_cluster_regions(cluster_result: ClusterResult, min_area: int) -> list[RegionStats]:
    """Regions found across every cluster except the largest.

    Shared by spot.py and stripe.py: the dominant (largest-fraction)
    cluster is treated as the fish's base body colour, since spots/stripes
    are by definition a minority colour against a base - including the
    dominant cluster would count the fish's own body outline as one giant
    region.

    Args:
        cluster_result: Output of clustering.assign_clusters().
        min_area: Passed through to find_regions() as the noise-filtering
            floor.

    Returns:
        All RegionStats across every non-dominant cluster's binary mask,
        pooled into one list (a region doesn't carry its source cluster
        index - callers needing that should call find_regions() directly
        per cluster instead).
    """
    if cluster_result.fractions.size == 0:
        return []
    dominant_cluster = int(cluster_result.fractions.argmax())
    regions = []
    for cluster_index in range(len(cluster_result.fractions)):
        if cluster_index == dominant_cluster:
            continue
        binary_mask = cluster_result.binary_mask_for_cluster(cluster_index)
        regions.extend(find_regions(binary_mask, min_area=min_area))
    return regions
