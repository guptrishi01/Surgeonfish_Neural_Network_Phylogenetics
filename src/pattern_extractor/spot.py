"""Spot/freckle-dimension features derived from cluster region shapes.

This project's own extension on top of patternize's colour-region output -
see __init__.py for how this differs from patternize itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pattern_extractor.clustering import ClusterResult
from pattern_extractor.config import RegionConfig, SpotConfig
from pattern_extractor.geometry import non_dominant_cluster_regions


@dataclass
class SpotFeatures:
    """Spot-dimension features for one image.

    Attributes:
        spot_count: Number of spot-like (small, roughly round, bounded
            area - see SpotConfig.max_spot_area_fraction) regions found
            across all non-dominant clusters.
        mean_spot_area: Mean pixel area of spot-like regions (0 if none).
        area_std: Standard deviation of spot-like region areas (0 if <2).
        spot_present: True if spot_count meets the configured minimum.
    """

    spot_count: int
    mean_spot_area: float
    area_std: float
    spot_present: bool


def extract_spot_features(
    cluster_result: ClusterResult,
    region_config: RegionConfig,
    spot_config: SpotConfig,
) -> SpotFeatures:
    """Finds spot-like regions across every non-dominant cluster.

    The dominant cluster (largest fraction) is treated as the fish's base
    body colour and excluded - spots are, by definition, a minority colour
    against a base, and including the dominant cluster would count the
    fish's own body outline as one giant "spot."

    Args:
        cluster_result: Output of clustering.assign_clusters().
        region_config: Shared region-analysis thresholds.
        spot_config: Spot-classification thresholds.

    Returns:
        SpotFeatures for this image.
    """
    regions = non_dominant_cluster_regions(cluster_result, region_config.min_region_area_fraction)
    total_masked_pixels = len(cluster_result.mask_coords[0])
    max_area = spot_config.max_spot_area_fraction * total_masked_pixels
    spot_areas = [
        region.area for region in regions
        if region.eccentricity <= spot_config.max_eccentricity_for_spot and region.area <= max_area
    ]

    spot_count = len(spot_areas)
    mean_area = float(np.mean(spot_areas)) if spot_count else 0.0
    area_std = float(np.std(spot_areas)) if spot_count > 1 else 0.0
    return SpotFeatures(
        spot_count=spot_count,
        mean_spot_area=mean_area,
        area_std=area_std,
        spot_present=spot_count >= spot_config.min_spot_count_for_presence,
    )
