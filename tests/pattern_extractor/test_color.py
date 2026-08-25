"""Unit tests for coloring-dimension feature extraction."""

from __future__ import annotations

import colorsys

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

    features = extract_color_features(image, mask, cluster_result, ColorConfig())

    assert features.is_solid is True
    assert features.hue_dispersion < 1e-9  # floating-point noise, not exactly 0


def test_lighting_gradient_on_one_real_color_is_still_classified_solid():
    # The actual bug this feature was rewritten to fix (v2.2.1): a real
    # photo's own lighting gradient on a genuinely solid-coloured fish
    # should not make it look multi-coloured. Same hue/saturation, four
    # brightness bands with per-pixel jitter (see test_clustering.py's
    # equivalent regression test for why a jitter-free version wouldn't be
    # a meaningful test of this).
    rng = np.random.default_rng(0)
    height, width = 60, 30
    image = np.zeros((height, width, 3), dtype=np.uint8)
    row_value = np.linspace(0.3, 0.9, height)
    for row in range(height):
        for col in range(width):
            value = np.clip(row_value[row] + rng.normal(0, 0.01), 0.0, 1.0)
            saturation = np.clip(0.8 + rng.normal(0, 0.01), 0.0, 1.0)
            r, g, b = colorsys.hsv_to_rgb(0.15, saturation, value)
            image[row, col] = (round(r * 255), round(g * 255), round(b * 255))
    mask = np.ones((height, width), dtype=bool)
    cluster_result = _cluster_result_for(image, mask, k=4)

    features = extract_color_features(image, mask, cluster_result, ColorConfig())

    assert features.is_solid is True


def test_evenly_split_two_color_image_is_not_solid():
    height, width = 40, 40
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :20] = (220, 20, 20)
    image[:, 20:] = (20, 20, 220)
    mask = np.ones((height, width), dtype=bool)
    cluster_result = _cluster_result_for(image, mask, k=2)

    features = extract_color_features(image, mask, cluster_result, ColorConfig())

    assert features.is_solid is False
    assert features.hue_dispersion > ColorConfig().max_hue_dispersion_for_solid


def test_n_significant_colors_counts_clusters_above_five_percent():
    height, width = 40, 40
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :20] = (220, 20, 20)
    image[:, 20:] = (20, 20, 220)
    mask = np.ones((height, width), dtype=bool)
    cluster_result = _cluster_result_for(image, mask, k=2)

    features = extract_color_features(image, mask, cluster_result, ColorConfig())

    assert features.n_significant_colors == 2


def test_palette_and_fractions_pass_through_from_cluster_result():
    height, width = 30, 30
    image = np.full((height, width, 3), (10, 20, 30), dtype=np.uint8)
    mask = np.ones((height, width), dtype=bool)
    cluster_result = _cluster_result_for(image, mask, k=2)

    features = extract_color_features(image, mask, cluster_result, ColorConfig())

    assert features.palette_rgb.shape[1] == 3
    assert features.fractions.shape[0] == features.palette_rgb.shape[0]


def test_empty_mask_does_not_crash():
    # fit_reference()/assign_clusters() already raise on a fully-empty mask
    # (pipeline.py catches that as ValueError and skips the image, per its
    # own docstring) - this tests extract_color_features's own guard
    # directly, via a ClusterResult that was never routed through
    # clustering at all, matching the pattern spot.py/stripe.py's own
    # empty-cluster-result tests use.
    from pattern_extractor.clustering import ClusterResult

    height, width = 10, 10
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    empty_result = ClusterResult(
        centers=np.zeros((0, 3)),
        labels=np.zeros((0,), dtype=int),
        fractions=np.zeros((0,)),
        mask_shape=(height, width),
        mask_coords=(np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)),
    )

    features = extract_color_features(image, mask, empty_result, ColorConfig())

    assert features.hue_dispersion == 0.0
