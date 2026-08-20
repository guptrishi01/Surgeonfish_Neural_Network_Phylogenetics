"""Unit tests for spot-dimension feature extraction."""

from __future__ import annotations

import numpy as np

from pattern_extractor.clustering import ClusterResult, assign_clusters, fit_reference
from pattern_extractor.config import ClusteringConfig, RegionConfig, SpotConfig
from pattern_extractor.spot import extract_spot_features


def _cluster_result_for(image: np.ndarray, mask: np.ndarray, k: int):
    clustering_config = ClusteringConfig(k=k, random_seed=0)
    reference_centers = fit_reference(image, mask, clustering_config)
    return assign_clusters(image, mask, reference_centers, clustering_config)


def _spotted_image(size=(60, 60)):
    height, width = size
    image = np.full((height, width, 3), (30, 150, 40), dtype=np.uint8)  # green base
    mask = np.ones((height, width), dtype=bool)
    centers = [(10, 10), (10, 40), (30, 25), (45, 12), (45, 45)]
    for cy, cx in centers:
        yy, xx = np.ogrid[:height, :width]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= 4**2
        image[blob] = (200, 30, 30)  # red spots
    return image, mask


def test_spotted_image_is_detected_as_spot_present():
    image, mask = _spotted_image()
    cluster_result = _cluster_result_for(image, mask, k=2)

    features = extract_spot_features(
        cluster_result, RegionConfig(min_region_area_px=5), SpotConfig()
    )

    assert features.spot_count >= 3
    assert features.spot_present is True
    assert features.mean_spot_area > 0


def test_solid_color_image_has_no_spots():
    height, width = 30, 30
    image = np.full((height, width, 3), (80, 80, 80), dtype=np.uint8)
    mask = np.ones((height, width), dtype=bool)
    cluster_result = _cluster_result_for(image, mask, k=2)

    features = extract_spot_features(cluster_result, RegionConfig(), SpotConfig())

    assert features.spot_count == 0
    assert features.spot_present is False


def test_empty_cluster_result_returns_no_spots():
    empty_result = ClusterResult(
        centers=np.zeros((0, 3)),
        labels=np.zeros((0,), dtype=int),
        fractions=np.zeros((0,)),
        mask_shape=(5, 5),
        mask_coords=(np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)),
    )

    features = extract_spot_features(empty_result, RegionConfig(), SpotConfig())

    assert features.spot_present is False
