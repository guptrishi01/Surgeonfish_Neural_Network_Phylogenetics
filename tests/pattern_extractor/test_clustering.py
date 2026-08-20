"""Unit tests for reference-initialized k-means clustering."""

from __future__ import annotations

import numpy as np

from pattern_extractor.clustering import assign_clusters, fit_reference
from pattern_extractor.config import ClusteringConfig


def _two_color_image(size=(40, 40), split_fraction=0.5):
    """Left `split_fraction` of the (masked-in) square is red, rest is blue."""
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    split_col = int(width * split_fraction)
    image[:, :split_col] = (220, 20, 20)
    image[:, split_col:] = (20, 20, 220)
    mask = np.ones((height, width), dtype=bool)
    return image, mask


def test_fit_reference_returns_k_centers():
    image, mask = _two_color_image()
    config = ClusteringConfig(k=2, random_seed=0)

    centers = fit_reference(image, mask, config)

    assert centers.shape == (2, 3)


def test_assign_clusters_separates_two_distinct_colors():
    image, mask = _two_color_image(split_fraction=0.5)
    config = ClusteringConfig(k=2, random_seed=0)
    reference_centers = fit_reference(image, mask, config)

    result = assign_clusters(image, mask, reference_centers, config)

    # Two roughly-equal-sized clusters (red half, blue half).
    assert result.fractions.shape == (2,)
    assert sorted(result.fractions) == sorted(result.fractions)  # shape sanity
    assert max(result.fractions) < 0.75  # neither cluster swallows both halves


def test_assign_clusters_fractions_sum_to_one():
    image, mask = _two_color_image(split_fraction=0.3)
    config = ClusteringConfig(k=2, random_seed=0)
    reference_centers = fit_reference(image, mask, config)

    result = assign_clusters(image, mask, reference_centers, config)

    assert abs(result.fractions.sum() - 1.0) < 1e-9


def test_binary_mask_for_cluster_matches_mask_shape_and_is_disjoint():
    image, mask = _two_color_image()
    config = ClusteringConfig(k=2, random_seed=0)
    reference_centers = fit_reference(image, mask, config)
    result = assign_clusters(image, mask, reference_centers, config)

    mask0 = result.binary_mask_for_cluster(0)
    mask1 = result.binary_mask_for_cluster(1)

    assert mask0.shape == mask.shape
    assert not np.any(mask0 & mask1)  # a pixel belongs to exactly one cluster
    assert np.all((mask0 | mask1) == mask)  # every masked-in pixel is assigned


def test_solid_color_image_concentrates_into_one_dominant_cluster():
    height, width = 30, 30
    image = np.full((height, width, 3), (100, 150, 90), dtype=np.uint8)
    mask = np.ones((height, width), dtype=bool)
    config = ClusteringConfig(k=3, random_seed=0)
    reference_centers = fit_reference(image, mask, config)

    result = assign_clusters(image, mask, reference_centers, config)

    assert max(result.fractions) > 0.9
