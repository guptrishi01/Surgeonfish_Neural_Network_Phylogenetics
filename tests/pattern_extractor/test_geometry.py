"""Unit tests for connected-component region shape analysis."""

from __future__ import annotations

import numpy as np

from pattern_extractor.geometry import find_regions


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
