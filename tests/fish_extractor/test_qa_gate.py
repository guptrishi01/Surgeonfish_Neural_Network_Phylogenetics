"""Unit tests for the accept/flag QA-gate decision logic."""

from __future__ import annotations

from fish_extractor.config import QAGateConfig
from fish_extractor.qa_gate import Box, evaluate

_CONFIG = QAGateConfig(
    min_confidence=0.35,
    max_center_offset_fraction=0.35,
    min_box_area_fraction=0.05,
    max_box_area_fraction=0.95,
)


def _centered_box(image_size: tuple[int, int], scale: float = 0.5, confidence: float = 0.8) -> Box:
    width, height = image_size
    box_w, box_h = width * scale, height * scale
    x0 = (width - box_w) / 2
    y0 = (height - box_h) / 2
    return Box(x0=x0, y0=y0, x1=x0 + box_w, y1=y0 + box_h, confidence=confidence)


def test_single_centered_confident_box_is_accepted():
    image_size = (1000, 800)
    result = evaluate([_centered_box(image_size)], image_size, _CONFIG)

    assert result.status == "accepted"
    assert result.reason == "ok"
    assert result.box is not None


def test_no_boxes_is_flagged_no_detection():
    result = evaluate([], (1000, 800), _CONFIG)

    assert result.status == "flagged"
    assert result.reason == "no_detection"
    assert result.box is None


def test_low_confidence_box_is_flagged_no_detection():
    image_size = (1000, 800)
    low_confidence_box = _centered_box(image_size, confidence=0.1)

    result = evaluate([low_confidence_box], image_size, _CONFIG)

    assert result.status == "flagged"
    assert result.reason == "no_detection"


def test_two_qualifying_boxes_is_flagged_multiple_fish():
    image_size = (1000, 800)
    boxes = [
        Box(x0=0, y0=0, x1=200, y1=200, confidence=0.8),
        Box(x0=700, y0=500, x1=900, y1=750, confidence=0.7),
    ]

    result = evaluate(boxes, image_size, _CONFIG)

    assert result.status == "flagged"
    assert result.reason == "multiple_fish"
    assert result.box is None


def test_tiny_box_is_flagged_box_too_small():
    image_size = (1000, 800)
    tiny_box = Box(x0=490, y0=390, x1=510, y1=410, confidence=0.9)

    result = evaluate([tiny_box], image_size, _CONFIG)

    assert result.status == "flagged"
    assert result.reason == "box_too_small"
    assert result.box is tiny_box


def test_frame_filling_box_is_flagged_box_too_large():
    image_size = (1000, 800)
    huge_box = Box(x0=0, y0=0, x1=1000, y1=800, confidence=0.9)

    result = evaluate([huge_box], image_size, _CONFIG)

    assert result.status == "flagged"
    assert result.reason == "box_too_large"


def test_off_center_box_is_flagged_off_center():
    image_size = (1000, 800)
    corner_box = Box(x0=0, y0=0, x1=300, y1=240, confidence=0.9)

    result = evaluate([corner_box], image_size, _CONFIG)

    assert result.status == "flagged"
    assert result.reason == "off_center"
    assert result.center_offset_fraction > _CONFIG.max_center_offset_fraction


def test_center_offset_fraction_is_reported_for_accepted_box():
    image_size = (1000, 800)

    result = evaluate([_centered_box(image_size)], image_size, _CONFIG)

    assert result.center_offset_fraction == 0.0


def test_near_duplicate_boxes_around_one_fish_are_deduplicated_and_accepted():
    # Two heavily-overlapping boxes (IoU ~0.84) around the same centered
    # object - a real Grounding DINO failure mode, not two fish.
    image_size = (1000, 800)
    boxes = [
        Box(x0=390, y0=290, x1=610, y1=510, confidence=0.9),
        Box(x0=400, y0=300, x1=620, y1=520, confidence=0.85),
    ]

    result = evaluate(boxes, image_size, _CONFIG)

    assert result.status == "accepted"
    assert result.box.confidence == 0.9  # the higher-confidence box of the pair was kept


def test_genuinely_separate_boxes_stay_flagged_multiple_fish():
    # Adjacent but mostly non-overlapping boxes (IoU ~0.14) - two real,
    # distinct fish should not be merged by deduplication.
    image_size = (1000, 800)
    boxes = [
        Box(x0=400, y0=300, x1=600, y1=500, confidence=0.9),
        Box(x0=550, y0=300, x1=750, y1=500, confidence=0.8),
    ]

    result = evaluate(boxes, image_size, _CONFIG)

    assert result.status == "flagged"
    assert result.reason == "multiple_fish"
