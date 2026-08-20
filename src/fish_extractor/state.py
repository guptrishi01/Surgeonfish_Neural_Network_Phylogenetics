"""Resumable per-image progress tracking for the fish-extraction pipeline.

Mirrors ``dataset_builder.state``: re-running the pipeline must not re-run
detection/segmentation on an image already processed in a prior run, and a
human's review-page decision (accept a flagged box, or exclude an image
entirely) must stick across re-runs rather than being re-litigated.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Terminal statuses: re-running the pipeline skips any image already in one
# of these states rather than re-running detection/segmentation on it.
_TERMINAL_STATUSES = frozenset({"accepted", "flagged", "excluded"})


@dataclass
class ImageExtractionState:
    """Progress for a single source image.

    Attributes:
        status: ``"pending"`` (not yet processed), ``"accepted"`` (QA gate
            passed, extraction written), ``"flagged"`` (needs human review),
            or ``"excluded"`` (a human permanently excluded this image via
            the review page).
        reason: Machine-readable reason code from the QA gate (e.g.
            ``"ok"``, ``"multiple_fish"``, ``"off_center"``), or
            ``"human_excluded"``.
        box: ``[x0, y0, x1, y1, confidence]`` of the box used, if any.
        center_offset_fraction: See ``qa_gate.QAResult``.
        mask_path: Path to the saved segmentation mask, if extraction ran.
        extracted_path: Path to the saved extracted-fish image, if
            extraction ran.
    """

    status: str = "pending"
    reason: str = ""
    box: list[float] | None = None
    center_offset_fraction: float | None = None
    mask_path: str | None = None
    extracted_path: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


class ExtractionState:
    """In-memory + on-disk store of every source image's ImageExtractionState."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._images: dict[str, ImageExtractionState] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        for key, entry in raw.items():
            self._images[key] = ImageExtractionState(
                status=entry.get("status", "pending"),
                reason=entry.get("reason", ""),
                box=entry.get("box"),
                center_offset_fraction=entry.get("center_offset_fraction"),
                mask_path=entry.get("mask_path"),
                extracted_path=entry.get("extracted_path"),
            )
        logger.info(
            "Loaded extraction state for %d image(s) from %s", len(self._images), self._path
        )

    def save(self) -> None:
        """Writes the full state to disk, overwriting any prior file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {key: asdict(state) for key, state in self._images.items()}
        self._path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    def get(self, image_key: str) -> ImageExtractionState:
        """Returns the state for an image, creating a fresh one if absent.

        Args:
            image_key: A stable identifier for the source image - the path
                relative to raw_images_root, e.g.
                ``"Acanthurus/Acanthurus_guttatus/000_reference.jpg"``.
        """
        if image_key not in self._images:
            self._images[image_key] = ImageExtractionState()
        return self._images[image_key]

    def is_processed(self, image_key: str) -> bool:
        """True if this image already has a terminal-status result on file."""
        return image_key in self._images and self._images[image_key].is_terminal

    def items(self) -> list[tuple[str, ImageExtractionState]]:
        """Every (image_key, state) pair currently on file, key-sorted."""
        return sorted(self._images.items())
