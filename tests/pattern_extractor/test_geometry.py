"""Unit tests for connected-component region shape analysis."""

from __future__ import annotations

import numpy as np

from pattern_extractor.clustering import ClusterResult
from pattern_extractor.geometry import find_regions, non_dominant_cluster_regions


def test_single_round_blob_has_low_eccentricity():
    mask = np.zeros((40, 40), dtype=bool)
    yy, xx = np.ogrid[:40, :40]
    circle = (yy - 20) ** 2 + (xx - 20) ** 2 <= 8**2
    mask[circle] = True

    regions = find_regions(mask, min_area=1)

    assert len(regions) == 1
    assert regions[0].eccentricity < 0.5


def test_thin_horizontal_line_has_high_eccentricity():
    mask = np.zeros((40, 40), dtype=bool)
    mask[20, 5:35] = True

    regions = find_regions(mask, min_area=1)

    assert len(regions) == 1
    assert regions[0].eccentricity > 0.9


def test_multiple_disjoint_blobs_are_separate_regions():
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:9, 5:9] = True
    mask[30:34, 30:34] = True

    regions = find_regions(mask, min_area=1)

    assert len(regions) == 2


def test_min_area_filters_small_noise_regions():
    mask = np.zeros((40, 40), dtype=bool)
    mask[5, 5] = True  # 1-pixel speck
    mask[20:30, 20:30] = True  # a real 100-pixel blob

    regions = find_regions(mask, min_area=5)

    assert len(regions) == 1
    assert regions[0].area == 100


def test_region_area_matches_pixel_count():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:15, 10:20] = True  # 5x10 = 50 pixels

    regions = find_regions(mask, min_area=1)

    assert regions[0].area == 50


def test_noise_floor_scales_with_total_masked_area_not_a_fixed_pixel_count():
    # A 50-pixel region would have cleared the old fixed 20px absolute
    # floor regardless of image size. At a real-scale total masked area
    # (100,000 pixels here), 0.1% of that is 100px - the 50-pixel region
    # should now be filtered as noise, which a fixed pixel count could
    # never do regardless of how the image scaled up.
    total_pixels = 100_000
    noise_size = 50
    dominant_size = total_pixels - noise_size

    # A single row: columns [0, dominant_size) are the dominant cluster,
    # the last `noise_size` columns form one contiguous non-dominant blob.
    labels = np.array([0] * dominant_size + [1] * noise_size)
    mask_coords = (np.zeros(total_pixels, dtype=int), np.arange(total_pixels))
    fractions = np.array([dominant_size / total_pixels, noise_size / total_pixels])
    cluster_result = ClusterResult(
        centers=np.zeros((2, 3)),
        labels=labels,
        fractions=fractions,
        mask_shape=(1, total_pixels),
        mask_coords=mask_coords,
    )

    regions = non_dominant_cluster_regions(cluster_result, min_area_fraction=0.001)

    assert regions == []  # the 50px noise blob was below the scaled floor (100px)


def test_noise_floor_keeps_regions_above_the_scaled_threshold():
    total_pixels = 100_000
    real_region_size = 500  # well above the 100px floor at 0.1% of 100,000
    dominant_size = total_pixels - real_region_size

    labels = np.array([0] * dominant_size + [1] * real_region_size)
    mask_coords = (np.zeros(total_pixels, dtype=int), np.arange(total_pixels))
    fractions = np.array([dominant_size / total_pixels, real_region_size / total_pixels])
    cluster_result = ClusterResult(
        centers=np.zeros((2, 3)),
        labels=labels,
        fractions=fractions,
        mask_shape=(1, total_pixels),
        mask_coords=mask_coords,
    )

    regions = non_dominant_cluster_regions(cluster_result, min_area_fraction=0.001)

    assert len(regions) == 1
    assert regions[0].area == real_region_size


def test_many_scattered_specks_dont_corrupt_a_real_blobs_centroid():
    # Exercises the bounding-box-offset path (find_objects()) with many
    # small components scattered across a large array, plus one real blob
    # positioned well away from the origin - a systematic offset bug in
    # translating local (within-bounding-box) coordinates back to global
    # ones would show up as a wrong centroid here.
    mask = np.zeros((500, 500), dtype=bool)
    rng = np.random.default_rng(0)
    ys, xs = rng.integers(0, 500, 200), rng.integers(0, 500, 200)
    mask[ys, xs] = True  # up to 200 single-pixel noise specks
    mask[300:310, 200:220] = True  # a real 10x20 = 200-pixel blob, off-origin

    regions = find_regions(mask, min_area=50)

    assert len(regions) == 1  # every noise speck (area 1) filtered by min_area
    assert regions[0].area == 200
    assert regions[0].centroid == (304.5, 209.5)  # exact center of rows 300-309, cols 200-219
