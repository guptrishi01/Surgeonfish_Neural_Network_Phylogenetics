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


def _spotted_image(size=(120, 120), rows=5, cols=5):
    # v2.2.2 calibrated min_spot_count_for_presence against real photos to
    # 12 (up from 3) - real solid-coloured fish routinely have a handful of
    # small round-ish noise regions (texture, JPEG artifacts), so a 5-spot
    # image no longer clears the bar. A genuinely freckled/spotted pattern
    # should have many more repeats than that anyway - a grid of spots
    # represents that more honestly than a borderline case tuned to pass.
    height, width = size
    image = np.full((height, width, 3), (30, 150, 40), dtype=np.uint8)  # green base
    mask = np.ones((height, width), dtype=bool)
    yy, xx = np.ogrid[:height, :width]
    row_positions = np.linspace(height * 0.15, height * 0.85, rows)
    col_positions = np.linspace(width * 0.15, width * 0.85, cols)
    for cy in row_positions:
        for cx in col_positions:
            blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= 4**2
            image[blob] = (200, 30, 30)  # red spots
    return image, mask


def test_spotted_image_is_detected_as_spot_present():
    image, mask = _spotted_image()
    cluster_result = _cluster_result_for(image, mask, k=2)

    features = extract_spot_features(cluster_result, RegionConfig(), SpotConfig())

    assert features.spot_count >= 12
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
