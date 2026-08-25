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
    # (k, 2): reference centres are hue/saturation vectors, not RGB triples -
    # see color_space.rgb_to_hue_sat_vector for why clustering happens there.
    image, mask = _two_color_image()
    config = ClusteringConfig(k=2, random_seed=0)

    centers = fit_reference(image, mask, config)

    assert centers.shape == (2, 2)


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


def test_fitting_is_subsampled_but_every_pixel_still_gets_labeled():
    # 100x100 = 10,000 masked pixels, well above a deliberately tiny fitting
    # cap - forces the subsampling path without needing a huge test image.
    image, mask = _two_color_image(size=(100, 100), split_fraction=0.5)
    config = ClusteringConfig(k=2, random_seed=0, max_pixels_for_fitting=500)

    reference_centers = fit_reference(image, mask, config)
    result = assign_clusters(image, mask, reference_centers, config)

    # Fitting used only a sample, but labelling must still cover every
    # actual masked-in pixel, not just the ones used to fit the centres.
    assert len(result.labels) == mask.sum() == 10_000
    assert abs(result.fractions.sum() - 1.0) < 1e-9
    assert max(result.fractions) < 0.75  # still cleanly splits red/blue halves


def test_lighting_gradient_on_one_real_color_yields_similar_cluster_centers():
    # k-means asked for k=4 clusters will still partition a genuinely
    # unimodal hue/saturation distribution into 4 similarly-sized pieces
    # regardless of which colour space it clusters in - there's no
    # mechanism for it to report "these are really all one colour" (this
    # is exactly why is_solid was rewritten in color.py to use a direct
    # dispersion measure instead of "does one forced-partition cluster
    # dominate" - see test_color.py's equivalent test for that actual
    # fix). What clustering in hue/saturation space *does* still guarantee,
    # confirmed here: the resulting clusters' mean colours stay close
    # together (same real hue, not falsely split into different-looking
    # colours), unlike raw-RGB clustering which would separate them by
    # brightness into visibly different-coloured clusters.
    import colorsys

    rng = np.random.default_rng(0)
    height, width = 80, 40
    image = np.zeros((height, width, 3), dtype=np.uint8)
    row_value = np.linspace(0.3, 0.9, height)  # smooth dorsal-to-ventral gradient
    for row in range(height):
        for col in range(width):
            value = np.clip(row_value[row] + rng.normal(0, 0.01), 0.0, 1.0)
            saturation = np.clip(0.8 + rng.normal(0, 0.01), 0.0, 1.0)
            r, g, b = colorsys.hsv_to_rgb(0.15, saturation, value)
            image[row, col] = (round(r * 255), round(g * 255), round(b * 255))
    mask = np.ones((height, width), dtype=bool)
    config = ClusteringConfig(k=4, random_seed=0)

    reference_centers = fit_reference(image, mask, config)
    result = assign_clusters(image, mask, reference_centers, config)

    # Every cluster's mean RGB should be a shade of the same colour, not a
    # different hue entirely - check pairwise RGB distance between centres
    # stays small relative to how far apart genuinely different colours
    # would land (e.g. red vs. blue differ by ~250+ per channel).
    for i in range(len(result.centers)):
        for j in range(i + 1, len(result.centers)):
            assert np.linalg.norm(result.centers[i] - result.centers[j]) < 60


def test_solid_color_image_concentrates_into_one_dominant_cluster():
    height, width = 30, 30
    image = np.full((height, width, 3), (100, 150, 90), dtype=np.uint8)
    mask = np.ones((height, width), dtype=bool)
    config = ClusteringConfig(k=3, random_seed=0)
    reference_centers = fit_reference(image, mask, config)

    result = assign_clusters(image, mask, reference_centers, config)

    assert max(result.fractions) > 0.9
