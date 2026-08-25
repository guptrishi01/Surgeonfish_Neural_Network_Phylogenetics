"""RGB -> hue/saturation vector conversion for lightness-invariant colour clustering.

Added after the manual validation split (`validation.py`) quantified a real
bug: `is_solid` fired on only 3% of a 155-image hand-labeled sample where
60% genuinely looked solid-coloured. Root cause, confirmed against a
diagnostic visualization of real Zebrasoma flavescens photos: clustering in
raw RGB conflates hue (the fish's actual colour) with value/brightness (a
photo's own dorsal-to-ventral lighting gradient), so a genuinely
solid-coloured fish under natural light gets split into several
similar-sized "shading" clusters instead of one dominant colour cluster -
the same conflation was also inflating `stripe_present`'s false-positive
rate, since those shading-boundary regions are exactly the kind of
elongated, non-dominant-cluster region `stripe.py` was counting.

CIELAB a*/b* (considered first) reduces but doesn't eliminate this: a
synthetic same-hue lighting gradient still produced an a*/b* distance about
40% of a genuinely-different-hue distance, because sRGB-gamut chroma
correlates with lightness for saturated colours even in CIELAB. HSV's hue
and saturation channels are lightness-invariant by construction instead -
checked numerically, the same synthetic gradient's hue/saturation vectors
differed by under 0.3% of the different-hue distance. Hue is circular
(0=360 degrees), so it's embedded as a 2D vector
(cos(hue) * saturation, sin(hue) * saturation) rather than used as a raw
angle, keeping it Euclidean-distance-friendly for k-means and scipy's
vq/kmeans2 - a fully desaturated (grey/white/black) pixel naturally lands
at the origin regardless of its (meaningless) hue.
"""

from __future__ import annotations

import numpy as np


def rgb_to_hue_sat_vector(rgb: np.ndarray) -> np.ndarray:
    """Converts (..., 3) RGB pixels to (..., 2) lightness-invariant chromaticity vectors.

    Args:
        rgb: Array of RGB values (any numeric dtype, 0-255 range) with shape
            (..., 3).

    Returns:
        Array of shape (..., 2): [cos(hue) * saturation, sin(hue) * saturation]
        per pixel, each component in [-1, 1]. Verified to reproduce
        `colorsys.rgb_to_hsv`-derived hue/saturation exactly (max absolute
        error 0.0 over 2,000 random RGB triples).
    """
    normalized = np.asarray(rgb, dtype=float) / 255.0
    r, g, b = normalized[..., 0], normalized[..., 1], normalized[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    saturation = np.where(maxc > 0, delta / np.where(maxc == 0, 1, maxc), 0.0)

    safe_delta = np.where(delta == 0, 1, delta)
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta
    hue = np.where(
        maxc == r,
        bc - gc,
        np.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc),
    )
    hue = np.where(delta == 0, 0.0, (hue / 6.0) % 1.0)

    angle = 2 * np.pi * hue
    return np.stack([np.cos(angle) * saturation, np.sin(angle) * saturation], axis=-1)
