"""Unit tests for stripe-dimension feature extraction."""

from __future__ import annotations

import numpy as np

from pattern_extractor.clustering import ClusterResult, assign_clusters, fit_reference
from pattern_extractor.config import ClusteringConfig, RegionConfig, StripeConfig
from pattern_extractor.stripe import _periodicity_strength, extract_stripe_features


def _cluster_result_for(image: np.ndarray, mask: np.ndarray, k: int):
    clustering_config = ClusteringConfig(k=k, random_seed=0)
    reference_centers = fit_reference(image, mask, clustering_config)
    return assign_clusters(image, mask, reference_centers, clustering_config)


def _striped_image(size=(60, 60), band_width=6):
    """Vertical alternating bands - a stripe pattern along the horizontal axis."""
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for col in range(width):
        band_index = col // band_width
        image[:, col] = (200, 30, 30) if band_index % 2 == 0 else (30, 30, 200)
    mask = np.ones((height, width), dtype=bool)
    return image, mask


def test_striped_image_has_high_periodicity_strength():
    image, mask = _striped_image()

    strength = _periodicity_strength(image, mask)

    assert strength > 0.15


def test_solid_image_has_near_zero_periodicity_strength():
    height, width = 40, 40
    image = np.full((height, width, 3), (90, 90, 90), dtype=np.uint8)
    mask = np.ones((height, width), dtype=bool)

    strength = _periodicity_strength(image, mask)

    assert strength == 0.0


def test_too_few_masked_pixels_returns_zero_periodicity():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    assert _periodicity_strength(image, mask) == 0.0


def test_striped_image_is_detected_as_stripe_present():
    image, mask = _striped_image()
    cluster_result = _cluster_result_for(image, mask, k=2)

    features = extract_stripe_features(
        image, mask, cluster_result, RegionConfig(), StripeConfig()
    )

    assert features.stripe_present is True


def test_solid_image_is_not_detected_as_stripe_present():
    height, width = 30, 30
    image = np.full((height, width, 3), (80, 80, 80), dtype=np.uint8)
    mask = np.ones((height, width), dtype=bool)
    cluster_result = _cluster_result_for(image, mask, k=2)

    features = extract_stripe_features(image, mask, cluster_result, RegionConfig(), StripeConfig())

    assert features.stripe_present is False
    assert features.elongated_region_count == 0


def test_empty_cluster_result_still_returns_valid_features():
    empty_result = ClusterResult(
        centers=np.zeros((0, 3)),
        labels=np.zeros((0,), dtype=int),
        fractions=np.zeros((0,)),
        mask_shape=(5, 5),
        mask_coords=(np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)),
    )
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=bool)

    features = extract_stripe_features(image, mask, empty_result, RegionConfig(), StripeConfig())

    assert features.elongated_region_count == 0
    assert features.stripe_present is False
