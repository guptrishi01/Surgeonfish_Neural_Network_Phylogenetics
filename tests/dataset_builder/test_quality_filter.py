"""Unit tests for the automated candidate-image pre-filters."""

from __future__ import annotations

import io

from PIL import Image

from dataset_builder.config import QualityFilterConfig
from dataset_builder.quality_filter import average_hash, evaluate, hamming_distance

CONFIG = QualityFilterConfig(
    min_short_side_px=200,
    min_aspect_ratio=0.4,
    duplicate_hash_hamming_threshold=6,
    max_border_edge_density=0.35,
)


def _image_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _solid_image(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color)


def test_rejects_unreadable_bytes():
    result = evaluate(b"not an image", [], CONFIG)
    assert not result.accepted
    assert result.reason == "unreadable"


def test_rejects_below_min_short_side():
    image = _solid_image((150, 150), (100, 150, 200))
    result = evaluate(_image_bytes(image), [], CONFIG)
    assert not result.accepted
    assert result.reason == "too_small"


def test_rejects_extreme_aspect_ratio():
    image = _solid_image((1000, 250), (100, 150, 200))  # aspect 0.25 < 0.4
    result = evaluate(_image_bytes(image), [], CONFIG)
    assert not result.accepted
    assert result.reason == "extreme_aspect_ratio"


def test_accepts_plain_solid_image_of_sufficient_size():
    image = _solid_image((400, 400), (100, 150, 200))
    result = evaluate(_image_bytes(image), [], CONFIG)
    assert result.accepted
    assert result.width == 400
    assert result.height == 400


def test_rejects_near_duplicate_of_already_accepted_image():
    image = _solid_image((400, 400), (100, 150, 200))
    first = evaluate(_image_bytes(image), [], CONFIG)
    assert first.accepted

    second = evaluate(_image_bytes(image), [first.average_hash], CONFIG)
    assert not second.accepted
    assert second.reason == "near_duplicate"


def _half_split_image(
    size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]
) -> Image.Image:
    # A flat solid-color image has zero variance, so its average hash is
    # degenerately 0 regardless of color - use a two-tone image instead so
    # the hash actually reflects content.
    image = Image.new("RGB", size, top)
    pixels = image.load()
    for x in range(size[0]):
        for y in range(size[1] // 2, size[1]):
            pixels[x, y] = bottom
    return image


def test_distinct_images_are_not_flagged_as_duplicates():
    a = _half_split_image((400, 400), (10, 10, 10), (240, 240, 240))
    b = _half_split_image((400, 400), (240, 240, 240), (10, 10, 10))
    first = evaluate(_image_bytes(a), [], CONFIG)
    second = evaluate(_image_bytes(b), [first.average_hash], CONFIG)
    assert second.accepted


def test_rejects_busy_border_background():
    # Checkerboard border, quiet center - high edge density at the border ring.
    image = Image.new("RGB", (400, 400), (200, 200, 200))
    pixels = image.load()
    for x in range(400):
        for y in range(400):
            in_border = x < 48 or x >= 352 or y < 48 or y >= 352
            if in_border and (x // 4 + y // 4) % 2 == 0:
                pixels[x, y] = (0, 0, 0)
    result = evaluate(_image_bytes(image), [], CONFIG)
    assert not result.accepted
    assert result.reason == "busy_background"


def test_hamming_distance_identical_hashes_is_zero():
    image = _solid_image((400, 400), (50, 60, 70))
    h = average_hash(image)
    assert hamming_distance(h, h) == 0
