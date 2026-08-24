"""Orchestrates the fish-identification/extraction pipeline end to end.

For each source image: unzip (idempotent) -> EXIF-rotate/flatten -> detect
-> QA-gate -> (if accepted) segment, mask-cutout, crop, save. Flagged images
are recorded but not extracted; a human resolves them via review.py. State
is checkpointed after every image so an interrupted run resumes cleanly.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from dataset_builder.archive import unzip_all
from fish_extractor.config import ExtractionConfig, PipelineConfig
from fish_extractor.detector import GroundingDinoDetector
from fish_extractor.qa_gate import Box, evaluate
from fish_extractor.segmenter import Sam2Segmenter
from fish_extractor.state import ExtractionState

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_METADATA_FIELDS = [
    "image_key", "status", "reason", "confidence",
    "center_offset_fraction", "extracted_path", "mask_path",
]


@dataclass
class ExtractionResult:
    """Outcome of processing one source image.

    Attributes:
        image_key: Path to the source image, relative to raw_images_root.
        status: ``"accepted"`` or ``"flagged"`` (see ImageExtractionState).
        reason: QA-gate reason code.
    """

    image_key: str
    status: str
    reason: str


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """EXIF-rotates and flattens transparency onto white, returning RGB."""
    image = ImageOps.exif_transpose(image) or image
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return image.convert("RGB")


def _mask_cutout_and_crop(
    image: Image.Image, mask: np.ndarray, config: ExtractionConfig
) -> tuple[Image.Image, np.ndarray]:
    """Replaces everything outside `mask` with background_fill, then crops.

    Returns both the cropped color image and the mask cropped to the same
    bounds, pixel-aligned - downstream stages (pattern_extractor) load both
    and need them to agree on shape.

    Raises:
        ValueError: If the mask is empty (should not happen for a mask
            produced from a QA-gate-accepted box, but guarded against
            defensively since it would otherwise crop to nothing).
    """
    array = np.array(image.convert("RGB"))
    array[~mask] = config.background_fill
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("segmentation mask is empty, cannot crop")

    margin = config.crop_margin_px
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, array.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, array.shape[1])
    return Image.fromarray(array[y0:y1, x0:x1]), mask[y0:y1, x0:x1]


class FishExtractorPipeline:
    """Resumable, per-image fish identification & extraction pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        detector: GroundingDinoDetector | None = None,
        segmenter: Sam2Segmenter | None = None,
    ) -> None:
        """Args:
            config: Pipeline configuration.
            detector: Overrides the default GroundingDinoDetector - tests
                inject a stub here so no GPU/model download is needed.
            segmenter: Overrides the default Sam2Segmenter - same reason.
        """
        self._config = config
        self._detector = detector or GroundingDinoDetector(config.detection)
        self._segmenter = segmenter or Sam2Segmenter(config.segmentation)
        self._state = ExtractionState(config.state_path)

    def run(self, species_filter: set[str] | None = None) -> list[ExtractionResult]:
        """Processes every not-yet-processed source image.

        Args:
            species_filter: If given, restricts the run to these species
                names (``"Genus species"``, space-separated).

        Returns:
            One ExtractionResult per image processed this run (images
            already terminal from a prior run are skipped and not
            included).
        """
        unzip_all(self._config.raw_images_root)
        results = []
        for image_path in self._iter_source_images(species_filter):
            image_key = self._image_key(image_path)
            if self._state.is_processed(image_key):
                continue
            result = self._process_one(image_path, image_key)
            results.append(result)
            self._state.save()
        return results

    def force_accept_flagged(self, image_keys: set[str] | None = None) -> list[ExtractionResult]:
        """Extracts flagged images a human reviewer has confirmed are actually usable.

        The review page only offers "exclude permanently" - leaving an image
        unchecked means "still flagged," not "accept it," so a reviewer who
        decides a flagged image is actually fine has no way to act on that
        decision through the normal review loop. This bypasses the QA gate
        for exactly that case, using the box already stored from the
        original detection (the one that failed a QA-gate threshold, not a
        fresh detection).

        Only usable for flagged images that have a stored box - QA-gate
        reasons ``off_center``, ``box_too_small``, and ``box_too_large`` all
        keep the box; ``no_detection`` and ``multiple_fish`` do not, since
        there is no single box to segment from in either case. Those are
        skipped and reported, not silently dropped - they need a fresh
        detection or a manually-specified box, not a bypass.

        Args:
            image_keys: If given, restricts to these specific image keys.
                Default: every currently-flagged image with a stored box.

        Returns:
            One ExtractionResult per image actually force-accepted.
        """
        unzip_all(self._config.raw_images_root)
        results = []
        skipped_no_box = []
        for image_key, entry in self._state.items():
            if entry.status != "flagged":
                continue
            if image_keys is not None and image_key not in image_keys:
                continue
            if entry.box is None:
                skipped_no_box.append(image_key)
                continue

            image_path = self._config.raw_images_root / image_key
            raw_image = Image.open(image_path)
            raw_image.load()
            image = _flatten_to_rgb(raw_image)
            box = Box(x0=entry.box[0], y0=entry.box[1], x1=entry.box[2], y1=entry.box[3],
                      confidence=entry.box[4])

            mask = self._segmenter.segment(image, box)
            extracted, cropped_mask = _mask_cutout_and_crop(image, mask, self._config.extraction)

            extracted_dir = self._config.extracted_root / Path(image_key).parent
            extracted_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(image_key).stem
            extracted_path = extracted_dir / f"{stem}.png"
            mask_path = extracted_dir / f"{stem}_mask.png"
            extracted.save(extracted_path)
            Image.fromarray((cropped_mask * 255).astype("uint8")).save(mask_path)

            entry.status = "accepted"
            entry.reason = "human_confirmed"
            entry.extracted_path = str(extracted_path)
            entry.mask_path = str(mask_path)
            self._write_metadata_row(image_key, entry)
            self._state.save()
            results.append(ExtractionResult(image_key, "accepted", "human_confirmed"))
            logger.info("%s: force-accepted (human_confirmed)", image_key)

        if skipped_no_box:
            logger.warning(
                "%d flagged image(s) have no stored box (no_detection/multiple_fish) and "
                "were not force-accepted: %s", len(skipped_no_box), skipped_no_box
            )
        return results

    def _iter_source_images(self, species_filter: set[str] | None):
        """Yields every image path under raw_images_root, genus/species-sorted."""
        root = self._config.raw_images_root
        for genus_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for species_dir in sorted(p for p in genus_dir.iterdir() if p.is_dir()):
                species_name = species_dir.name.replace("_", " ")
                if species_filter and species_name not in species_filter:
                    continue
                for image_path in sorted(species_dir.glob("*")):
                    if image_path.suffix.lower() in _IMAGE_EXTENSIONS:
                        yield image_path

    def _image_key(self, image_path: Path) -> str:
        """Stable state/CSV identifier: image_path relative to raw_images_root."""
        return str(image_path.relative_to(self._config.raw_images_root)).replace("\\", "/")

    def _process_one(self, image_path: Path, image_key: str) -> ExtractionResult:
        """Runs one image through open -> detect -> QA-gate -> segment -> crop -> save.

        Never raises for an unreadable or ambiguous image - those are
        recorded as "flagged" (or "unreadable", still non-fatal) in state
        and the metadata CSV, so a single bad image can't abort the run.
        """
        entry = self._state.get(image_key)
        try:
            raw_image = Image.open(image_path)
            raw_image.load()
        except Exception:
            logger.exception("Unreadable image %s, flagging", image_key)
            entry.status, entry.reason = "flagged", "unreadable"
            self._write_metadata_row(image_key, entry)
            return ExtractionResult(image_key, "flagged", "unreadable")

        image = _flatten_to_rgb(raw_image)
        boxes = self._detector.detect(image)
        qa = evaluate(boxes, image.size, self._config.qa_gate)

        entry.reason = qa.reason
        entry.center_offset_fraction = qa.center_offset_fraction
        entry.box = (
            [qa.box.x0, qa.box.y0, qa.box.x1, qa.box.y1, qa.box.confidence] if qa.box else None
        )

        if qa.status == "flagged":
            entry.status = "flagged"
            self._write_metadata_row(image_key, entry)
            logger.info("%s: flagged (%s)", image_key, qa.reason)
            return ExtractionResult(image_key, "flagged", qa.reason)

        mask = self._segmenter.segment(image, qa.box)
        extracted, cropped_mask = _mask_cutout_and_crop(image, mask, self._config.extraction)

        extracted_dir = self._config.extracted_root / Path(image_key).parent
        extracted_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_key).stem
        extracted_path = extracted_dir / f"{stem}.png"
        mask_path = extracted_dir / f"{stem}_mask.png"
        extracted.save(extracted_path)
        Image.fromarray((cropped_mask * 255).astype("uint8")).save(mask_path)

        entry.status = "accepted"
        entry.extracted_path = str(extracted_path)
        entry.mask_path = str(mask_path)
        self._write_metadata_row(image_key, entry)
        logger.info("%s: accepted", image_key)
        return ExtractionResult(image_key, "accepted", "ok")

    def _write_metadata_row(self, image_key: str, entry) -> None:
        """Appends one row (writing the header first if the file is new)."""
        path = self._config.metadata_csv_path
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_METADATA_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "image_key": image_key,
                "status": entry.status,
                "reason": entry.reason,
                "confidence": entry.box[4] if entry.box else "",
                "center_offset_fraction": (
                    entry.center_offset_fraction if entry.center_offset_fraction is not None else ""
                ),
                "extracted_path": entry.extracted_path or "",
                "mask_path": entry.mask_path or "",
            })
