"""Unit tests for RGB -> hue/saturation vector conversion."""

from __future__ import annotations

import colorsys

import numpy as np

from pattern_extractor.color_space import rgb_to_hue_sat_vector


def test_matches_colorsys_reference_over_random_colors():
    rng = np.random.default_rng(0)
    samples = rng.integers(0, 256, size=(500, 3))

    vectors = rgb_to_hue_sat_vector(samples)

    for rgb, vector in zip(samples, vectors):
        r, g, b = (c / 255 for c in rgb)
        h, s, _v = colorsys.rgb_to_hsv(r, g, b)
        expected = np.array([np.cos(2 * np.pi * h) * s, np.sin(2 * np.pi * h) * s])
        assert np.allclose(vector, expected, atol=1e-9)


def test_grey_pixels_land_at_the_origin_regardless_of_shade():
    greys = np.array([[0, 0, 0], [80, 80, 80], [160, 160, 160], [255, 255, 255]])

    vectors = rgb_to_hue_sat_vector(greys)

    assert np.allclose(vectors, 0.0)


def test_same_hue_different_brightness_stays_close_together():
    # Same hue/saturation (a lighting gradient on one real colour), four
    # brightness levels - this is the property the whole module exists for:
    # a photographed fish's own dorsal-to-ventral shading shouldn't look
    # like a different colour to the clustering step.
    dark = np.array([76, 70, 15])
    bright = np.array([230, 211, 46])
    other_hue = np.array([40, 60, 200])  # a genuinely different colour

    dark_vec = rgb_to_hue_sat_vector(dark)
    bright_vec = rgb_to_hue_sat_vector(bright)
    other_vec = rgb_to_hue_sat_vector(other_hue)

    same_hue_distance = np.linalg.norm(dark_vec - bright_vec)
    different_hue_distance = np.linalg.norm(dark_vec - other_vec)

    assert same_hue_distance < different_hue_distance / 10


def test_output_shape_matches_input_leading_dimensions():
    rgb = np.zeros((5, 7, 3), dtype=np.uint8)

    vectors = rgb_to_hue_sat_vector(rgb)

    assert vectors.shape == (5, 7, 2)
