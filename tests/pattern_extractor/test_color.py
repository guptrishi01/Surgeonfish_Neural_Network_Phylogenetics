"""Unit tests for coloring-dimension feature extraction."""

from __future__ import annotations

import numpy as np

from pattern_extractor.clustering import assign_clusters, fit_reference
from pattern_extractor.color import extract_color_features
from pattern_extractor.config import ClusteringConfig, ColorConfig


def _cluster_result_for(image: np.ndarray, mask: np.ndarray, k: int):
    clustering_config = ClusteringConfig(k=k, random_seed=0)
    reference_centers = fit_reference(image, mask, clustering_config)
    return assign_clusters(image, mask, reference_centers, clustering_config)


def test_solid_color_image_is_classified_solid():
    height, width = 30, 30
    image = np.full((height, width, 3), (100, 150, 90), dtype=np.uint8)
    mask = np.ones((height, width), dtype=bool)
    cluster_result = _cluster_result_for(image, mask, k=3)

    features = extract_color_features(cluster_result, ColorConfig())

    assert features.is_solid is True
    assert features.dominant_fraction > 0.85


def test_evenly_split_two_color_image_is_not_solid():
    height, width = 40, 40
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :20] = (220, 20, 20)
    image[:, 20:] = (20, 20, 220)
    mask = np.ones((height, width), dtype=bool)
    cluster_result = _cluster_result_for(image, mask, k=2)

    features = extract_color_features(cluster_result, ColorConfig())

    assert features.is_solid is False
    assert features.dominant_fraction < ColorConfig().solid_dominant_fraction_threshold


def test_n_significant_colors_counts_clusters_above_five_percent():
    height, width = 40, 40
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :20] = (220, 20, 20)
    image[:, 20:] = (20, 20, 220)
    mask = np.ones((height, width), dtype=bool)
    cluster_result = _cluster_result_for(image, mask, k=2)

    features = extract_color_features(cluster_result, ColorConfig())

    assert features.n_significant_colors == 2


def test_palette_and_fractions_pass_through_from_cluster_result():
    height, width = 30, 30
    image = np.full((height, width, 3), (10, 20, 30), dtype=np.uint8)
    mask = np.ones((height, width), dtype=bool)
    cluster_result = _cluster_result_for(image, mask, k=2)

    features = extract_color_features(cluster_result, ColorConfig())

    assert features.palette_rgb.shape[1] == 3
    assert features.fractions.shape[0] == features.palette_rgb.shape[0]
