"""Pure accept/flag decision logic for a single image's detection results.

Deliberately has no dependency on torch/transformers/PIL, so it is fully
unit-testable with synthetic box lists - the only place a real judgment call
("is this one clean, centered fish") gets made mechanically, everything else
in this package is plumbing around it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fish_extractor.config import QAGateConfig


@dataclass
class Box:
    """A single detected bounding box, in pixel coordinates.

    Attributes:
        x0: Left edge.
        y0: Top edge.
        x1: Right edge.
        y1: Bottom edge.
        confidence: Detector confidence score for this box, in [0, 1].
    """

    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        """(x, y) pixel coordinates of the box's center."""
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    @property
    def area(self) -> float:
        """Box area in square pixels (0 if width/height is degenerate)."""
        return max(self.width, 0.0) * max(self.height, 0.0)


@dataclass
class QAResult:
    """Outcome of evaluating one image's detections against the QA gate.

    Attributes:
        status: ``"accepted"`` or ``"flagged"``.
        reason: Machine-readable reason code. ``"ok"`` when accepted;
            otherwise one of ``"no_detection"``, ``"multiple_fish"``,
            ``"box_too_small"``, ``"box_too_large"``, ``"off_center"``.
        box: The single qualifying box, if there is exactly one above
            ``min_confidence`` - set even when flagged for a
            size/centering reason, so a reviewer can see what was found.
        center_offset_fraction: Distance between the box's center and the
            image's center, as a fraction of the image's half-diagonal.
            ``None`` when there is no single qualifying box to measure.
    """

    status: str
    reason: str
    box: Box | None
    center_offset_fraction: float | None


def evaluate(
    boxes: list[Box],
    image_size: tuple[int, int],
    config: QAGateConfig,
) -> QAResult:
    """Decides whether a detection set shows one clean, centered fish.

    Args:
        boxes: All boxes the detector returned for this image (already
            filtered to the "fish" text prompt, but not yet by confidence).
        image_size: ``(width, height)`` of the source image in pixels.
        config: Auto-accept thresholds.

    Returns:
        A QAResult with status "accepted" only when exactly one box clears
        the confidence threshold and its size/position both pass; every
        other case is "flagged", never silently rejected.
    """
    qualifying = [b for b in boxes if b.confidence >= config.min_confidence]

    if len(qualifying) == 0:
        return QAResult(
            status="flagged", reason="no_detection", box=None, center_offset_fraction=None
        )
    if len(qualifying) > 1:
        return QAResult(
            status="flagged", reason="multiple_fish", box=None, center_offset_fraction=None
        )

    box = qualifying[0]
    width, height = image_size
    image_area = width * height
    area_fraction = box.area / image_area if image_area > 0 else 0.0

    box_cx, box_cy = box.center
    image_cx, image_cy = width / 2, height / 2
    half_diagonal = math.hypot(width, height) / 2
    offset_fraction = (
        math.hypot(box_cx - image_cx, box_cy - image_cy) / half_diagonal
        if half_diagonal > 0
        else 0.0
    )

    if area_fraction < config.min_box_area_fraction:
        return QAResult(
            status="flagged", reason="box_too_small",
            box=box, center_offset_fraction=offset_fraction,
        )
    if area_fraction > config.max_box_area_fraction:
        return QAResult(
            status="flagged", reason="box_too_large",
            box=box, center_offset_fraction=offset_fraction,
        )
    if offset_fraction > config.max_center_offset_fraction:
        return QAResult(
            status="flagged", reason="off_center", box=box, center_offset_fraction=offset_fraction
        )

    return QAResult(status="accepted", reason="ok", box=box, center_offset_fraction=offset_fraction)
