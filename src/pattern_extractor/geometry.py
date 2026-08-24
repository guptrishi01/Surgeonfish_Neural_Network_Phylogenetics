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
        minor_axis_length: Full width of the region's minor axis
            (``4 * sqrt(lambda2)``, the standard image-moment ellipse
            convention - same formula skimage's regionprops uses). A wide
            region can still score a high eccentricity if it's long enough
            (e.g. one smooth shading half spanning most of a fish's body,
            confirmed via a diagnostic visualization against real Phase 2
            output - see StripeConfig.max_stripe_width_fraction), so
            eccentricity alone doesn't distinguish "elongated" from
            "narrow." This field lets callers require both.
    """

    area: int
    centroid: tuple[float, float]
    eccentricity: float
    orientation_rad: float
    minor_axis_length: float


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
    if n == 0:
        return []
    # find_objects() locates each label's bounding box in a single pass
    # over the whole array; without it, a per-label `np.where(labeled ==
    # label_id)` re-scans the *entire* image for every component. A clean
    # synthetic test image has 1-2 components and never exposed this, but a
    # real photo's non-dominant colour cluster can have hundreds to
    # thousands of tiny noise/texture components (JPEG artifacts, water
    # texture, background flecks) against a multi-megapixel crop -
    # measured at ~50s for one such image with the naive approach, ~0.1s
    # with this one.
    bounding_boxes = ndimage.find_objects(labeled)
    regions = []
    for label_id, bbox in enumerate(bounding_boxes, start=1):
        if bbox is None:
            continue
        local_rows, local_cols = np.where(labeled[bbox] == label_id)
        area = len(local_rows)
        if area < min_area:
            continue
        rows = local_rows + bbox[0].start
        cols = local_cols + bbox[1].start
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
        minor_axis_length = 4 * math.sqrt(max(0.0, lambda2))
        regions.append(
            RegionStats(
                area=area,
                centroid=(row_mean, col_mean),
                eccentricity=eccentricity,
                orientation_rad=orientation,
                minor_axis_length=minor_axis_length,
            )
        )
    return regions


def non_dominant_cluster_regions(
    cluster_result: ClusterResult, min_area_fraction: float
) -> list[RegionStats]:
    """Regions found across every cluster except the largest.

    Shared by spot.py and stripe.py: the dominant (largest-fraction)
    cluster is treated as the fish's base body colour, since spots/stripes
    are by definition a minority colour against a base - including the
    dominant cluster would count the fish's own body outline as one giant
    region.

    Args:
        cluster_result: Output of clustering.assign_clusters().
        min_area_fraction: Regions smaller than this fraction of the
            image's *total* masked-in (fish) pixel count - not just the
            fraction of whichever single cluster a region belongs to - are
            dropped as noise. Resolved to an absolute pixel count here
            before being passed to find_regions(), so the noise floor
            scales with the actual crop size rather than staying fixed
            regardless of it.

    Returns:
        All RegionStats across every non-dominant cluster's binary mask,
        pooled into one list (a region doesn't carry its source cluster
        index - callers needing that should call find_regions() directly
        per cluster instead).
    """
    if cluster_result.fractions.size == 0:
        return []
    total_masked_pixels = len(cluster_result.mask_coords[0])
    min_area = max(1, round(total_masked_pixels * min_area_fraction))
    dominant_cluster = int(cluster_result.fractions.argmax())
    regions = []
    for cluster_index in range(len(cluster_result.fractions)):
        if cluster_index == dominant_cluster:
            continue
        binary_mask = cluster_result.binary_mask_for_cluster(cluster_index)
        regions.extend(find_regions(binary_mask, min_area=min_area))
    return regions
