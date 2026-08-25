"""Stripe-dimension features: region elongation plus FFT periodicity.

This project's own extension on top of patternize's colour-region output -
see __init__.py for how this differs from patternize itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pattern_extractor.clustering import ClusterResult
from pattern_extractor.config import RegionConfig, StripeConfig
from pattern_extractor.geometry import non_dominant_cluster_regions


@dataclass
class StripeFeatures:
    """Stripe-dimension features for one image.

    Attributes:
        elongated_region_count: Number of regions found across all
            non-dominant clusters that are both elongated (high
            eccentricity) and narrow (bounded minor-axis width relative to
            the fish's size) - see StripeConfig.max_stripe_width_fraction
            for why both checks are needed, not eccentricity alone.
        periodicity_strength: Normalized FFT peak strength of the fish
            region's intensity profile along its principal axis - high for
            a regularly repeating pattern (stripes), low for one solid
            block of colour or scattered spots.
        stripe_present: True if either signal clears its configured
            threshold - region-shape and periodicity are independent
            checks, since a single narrow elongated band (one region) can
            still be a real stripe pattern without strong periodicity.
    """

    elongated_region_count: int
    periodicity_strength: float
    stripe_present: bool


def _axis_periodicity(
    positions: np.ndarray, values: np.ndarray, length: int, min_cycles: int
) -> float:
    """FFT peak strength of `values` averaged into bins along one axis.

    Only frequencies representing at least `min_cycles` repeats across the
    profile are eligible to be the "peak" - see min_cycles's caller
    (StripeConfig.min_periodicity_cycles) for why: a single smooth colour
    gradient (one lobe, no repetition) concentrates almost all of its
    non-DC spectral energy in the lowest available frequency exactly like
    genuine periodicity does, so without this restriction the two are
    indistinguishable - confirmed both against real photos (a diagnostic
    visualization showed solid-coloured Zebrasoma species, which have a
    real dorsal-to-ventral lighting gradient, scoring *higher* periodicity
    than genuinely striped Acanthurus lineatus) and numerically (a
    synthetic single-lobe gradient scored 0.673 pre-fix vs. 0.269 for a
    genuine 10-stripe pattern - backwards).
    """
    sums = np.bincount(positions, weights=values, minlength=length)
    counts = np.bincount(positions, minlength=length)
    occupied = counts > 0
    if occupied.sum() < 4:
        return 0.0
    profile = sums[occupied] / counts[occupied]
    if np.allclose(profile, profile[0]):
        return 0.0

    profile = profile - profile.mean()
    spectrum = np.abs(np.fft.rfft(profile))[1:]  # drop the DC component
    if spectrum.size == 0 or spectrum.sum() == 0:
        return 0.0
    candidate = spectrum[min_cycles - 1 :]  # spectrum[0] = 1 cycle across the profile
    if candidate.size == 0:
        return 0.0
    return float(candidate.max() / spectrum.sum())


def _periodicity_strength(image_rgb: np.ndarray, mask: np.ndarray, min_cycles: int = 3) -> float:
    """Normalized FFT peak strength of the masked region's colour profile.

    Averages each colour channel's masked-in pixels into per-row and
    per-column bins, and measures how much of each 1D profile's non-DC
    variance concentrates in a plausible repeating-pattern frequency (at
    least `min_cycles` repeats across the profile - see `_axis_periodicity`
    for why the low-frequency band is excluded) - a proxy for "does colour
    repeat regularly along the body," which a stripe pattern does and a
    solid colour, a lighting gradient, or scattered spots don't. Checked
    per-channel (R, G, B) rather than collapsed to grayscale first: two
    stripe colours can have the same luminance-average (e.g. red vs. blue
    both average to the same grayscale value) while still alternating
    clearly in one channel, and a grayscale-only profile would miss that.
    Also checked along both image axes, rather than the mask's principal
    axis, since a roughly-square crop leaves that axis poorly defined. The
    strongest signal found across all channel/axis combinations is
    returned.

    Args:
        image_rgb: (H, W, 3) uint8 RGB array.
        mask: (H, W) boolean array, True for fish (masked-in) pixels.
        min_cycles: Minimum number of repeats across the profile for a
            frequency to be eligible as the "peak" - see StripeConfig.
            min_periodicity_cycles for the reasoning and default. Defaults
            to that same default so callers that don't have a StripeConfig
            handy (e.g. direct unit tests) still get the fixed behaviour.

    Returns:
        A value in [0, 1]; 0 for too few pixels or a uniform region.
    """
    ys, xs = np.where(mask)
    if ys.size < 4:
        return 0.0

    height, width = mask.shape
    best = 0.0
    for channel in range(image_rgb.shape[-1]):
        values = image_rgb[ys, xs, channel].astype(float)
        best = max(best, _axis_periodicity(ys, values, height, min_cycles))
        best = max(best, _axis_periodicity(xs, values, width, min_cycles))
    return best


def extract_stripe_features(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    cluster_result: ClusterResult,
    region_config: RegionConfig,
    stripe_config: StripeConfig,
) -> StripeFeatures:
    """Finds elongated regions and measures body-axis colour periodicity.

    Args:
        image_rgb: (H, W, 3) uint8 RGB array (the image clustering ran on).
        mask: (H, W) boolean array, True for fish (masked-in) pixels.
        cluster_result: Output of clustering.assign_clusters().
        region_config: Shared region-analysis thresholds.
        stripe_config: Stripe-classification thresholds.

    Returns:
        StripeFeatures for this image.
    """
    regions = non_dominant_cluster_regions(cluster_result, region_config.min_region_area_fraction)
    total_masked_pixels = len(cluster_result.mask_coords[0])
    max_width = stripe_config.max_stripe_width_fraction * math.sqrt(total_masked_pixels)
    elongated_count = sum(
        1
        for region in regions
        if region.eccentricity >= stripe_config.min_eccentricity_for_stripe
        and region.minor_axis_length <= max_width
    )

    periodicity = _periodicity_strength(image_rgb, mask, stripe_config.min_periodicity_cycles)
    stripe_present = (
        elongated_count >= stripe_config.min_elongated_region_count
        or periodicity >= stripe_config.min_periodicity_strength
    )
    return StripeFeatures(
        elongated_region_count=elongated_count,
        periodicity_strength=periodicity,
        stripe_present=stripe_present,
    )
