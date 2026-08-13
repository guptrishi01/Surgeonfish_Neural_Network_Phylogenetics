"""Automated pre-filtering for scraped candidate images.

These checks are a coarse first pass only - a cheap way to reject broken
files, near-duplicates, and obviously busy/cluttered backgrounds before a
human spot-checks a sample. They are not a substitute for that visual
review; thresholds here were tuned against a pilot batch, not derived
theoretically.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from dataset_builder.config import QualityFilterConfig

logger = logging.getLogger(__name__)

_HASH_SIZE = 8  # 8x8 -> 64-bit average hash
_EDGE_THRESHOLD = 40  # 0-255 gradient magnitude considered an "edge" pixel
_BORDER_FRACTION = 0.12  # width of the outer ring examined for background


@dataclass
class QualityResult:
    """Outcome of evaluating one candidate image.

    Attributes:
        accepted: Whether the image passed every hard filter.
        reason: Rejection reason (empty string if accepted).
        width: Image width in pixels.
        height: Image height in pixels.
        average_hash: 64-bit perceptual hash, for cross-candidate dedup.
        border_edge_density: Fraction of edge-classified pixels in the
            outer border ring; higher suggests a busier background.
    """

    accepted: bool
    reason: str
    width: int
    height: int
    average_hash: int
    border_edge_density: float


def average_hash(image: Image.Image) -> int:
    """Computes a 64-bit average hash for near-duplicate detection."""
    small = image.convert("L").resize(
        (_HASH_SIZE, _HASH_SIZE), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(small, dtype=np.float64)
    mean = pixels.mean()
    bits = (pixels > mean).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Counts differing bits between two average hashes."""
    return bin(hash_a ^ hash_b).count("1")


def _border_edge_density(image: Image.Image) -> float:
    """Estimates background busyness from edge density in the outer ring.

    A fish photographed close-up against open water or substrate should
    have a comparatively quiet border; reef clutter, other fish, or divers
    at the frame edges raise this value.
    """
    grayscale = image.convert("L")
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edges, dtype=np.uint8)
    h, w = arr.shape
    band_h = max(1, int(h * _BORDER_FRACTION))
    band_w = max(1, int(w * _BORDER_FRACTION))

    border_mask = np.zeros_like(arr, dtype=bool)
    border_mask[:band_h, :] = True
    border_mask[-band_h:, :] = True
    border_mask[:, :band_w] = True
    border_mask[:, -band_w:] = True

    border_pixels = arr[border_mask]
    if border_pixels.size == 0:
        return 0.0
    return float((border_pixels > _EDGE_THRESHOLD).mean())


def evaluate(
    image_bytes: bytes,
    accepted_hashes: list[int],
    config: QualityFilterConfig,
) -> QualityResult:
    """Runs every hard filter against one candidate image.

    Args:
        image_bytes: Raw downloaded image content.
        accepted_hashes: Average hashes of images already accepted for this
            species, checked for near-duplicates.
        config: Thresholds to apply.

    Returns:
        A QualityResult describing whether the image was accepted and why
        not, when rejected.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as exc:  # noqa: BLE001 - any decode failure is a reject
        logger.debug("Unreadable image: %s", exc)
        return QualityResult(False, "unreadable", 0, 0, 0, 0.0)

    width, height = image.size
    short_side = min(width, height)
    if short_side < config.min_short_side_px:
        return QualityResult(
            False, "too_small", width, height, 0, 0.0
        )

    aspect = min(width, height) / max(width, height)
    if aspect < config.min_aspect_ratio:
        return QualityResult(
            False, "extreme_aspect_ratio", width, height, 0, 0.0
        )

    image_hash = average_hash(image)
    for existing in accepted_hashes:
        if hamming_distance(image_hash, existing) <= config.duplicate_hash_hamming_threshold:
            return QualityResult(
                False, "near_duplicate", width, height, image_hash, 0.0
            )

    edge_density = _border_edge_density(image)
    if edge_density > config.max_border_edge_density:
        return QualityResult(
            False, "busy_background", width, height, image_hash, edge_density
        )

    return QualityResult(True, "", width, height, image_hash, edge_density)
