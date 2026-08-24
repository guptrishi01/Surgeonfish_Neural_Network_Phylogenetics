"""Configuration objects for the fish-identification/extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DetectionConfig:
    """Settings for zero-shot, text-prompted fish detection.

    Attributes:
        model_id: Hugging Face model id for a Grounding DINO checkpoint,
            loaded via ``transformers.AutoModelForZeroShotObjectDetection``
            (not the original IDEA-Research repo, which requires compiling a
            custom CUDA op).
        text_prompt: Open-vocabulary query. Grounding DINO expects
            lowercase, period-terminated phrases.
        box_threshold: Minimum detection confidence for a box to be
            considered at all.
        text_threshold: Minimum per-token text-match confidence used to
            decide whether a box actually matches ``text_prompt``.
    """

    model_id: str = "IDEA-Research/grounding-dino-tiny"
    text_prompt: str = "fish."
    box_threshold: float = 0.35
    text_threshold: float = 0.25


@dataclass
class SegmentationConfig:
    """Settings for turning a detected box into a precise mask.

    Attributes:
        model_id: Hugging Face model id (or local checkpoint name) for a
            SAM 2.1 model.
    """

    model_id: str = "facebook/sam2.1-hiera-large"


@dataclass
class QAGateConfig:
    """Thresholds deciding whether a detection auto-accepts or gets flagged.

    A detection auto-accepts only when all of these hold; otherwise it is
    flagged for human review rather than silently accepted or dropped, per
    CLAUDE.md's "every gate failure blocks for an explicit decision"
    convention.

    Attributes:
        min_confidence: The single qualifying box's detection confidence
            must be at least this high.
        max_center_offset_fraction: Maximum allowed distance between the
            box's center and the image's center, expressed as a fraction of
            the image's half-diagonal (0 = dead center, 1 = box center sits
            on a corner).
        min_box_area_fraction: Minimum box area as a fraction of the full
            image area (rules out a fish that's a tiny, distant speck).
        max_box_area_fraction: Maximum box area as a fraction of the full
            image area (rules out a box that just covers the whole frame,
            which is rarely a clean single-fish detection).
        duplicate_iou_threshold: Two qualifying boxes with IoU (intersection
            over union) at or above this are treated as one detection - the
            higher-confidence box is kept - rather than as two fish. Guards
            against a real Grounding DINO failure mode: near-duplicate boxes
            around the same object, which would otherwise be
            indistinguishable from a genuine multi-fish photo and flag every
            such image as "multiple_fish". 0.5 is a standard NMS-style
            overlap threshold (used by, e.g., torchvision's default IoU cut
            for duplicate suppression) - high enough that two boxes around
            genuinely different, non-overlapping fish won't be merged.
        min_secondary_box_area_ratio: A qualifying box smaller than this
            fraction of the *largest* qualifying box's area is dropped as
            background clutter rather than counted as a second fish.
            Empirically motivated: inspecting real "multiple_fish" flags
            that survived deduplication showed a consistent pattern of one
            large, high-confidence box (the actual subject, 20-30% of the
            image) plus a small, non-overlapping, lower-confidence box
            (under 14% of the *main box's* area in every case checked) -
            almost certainly background clutter (a distant fish, a fin,
            reef texture), not a genuine second subject. A real second fish
            is typically comparable in size to the first, not a sliver of
            it. 0.2 sits well above the observed clutter ratios while still
            treating two similarly-sized boxes as two real fish.
    """

    min_confidence: float = 0.35
    max_center_offset_fraction: float = 0.35
    min_box_area_fraction: float = 0.05
    max_box_area_fraction: float = 0.95
    duplicate_iou_threshold: float = 0.5
    min_secondary_box_area_ratio: float = 0.2


@dataclass
class ExtractionConfig:
    """Settings for turning an accepted mask into the saved output image.

    No resize/pad step: the old pipeline standardized crops to a fixed
    1024x1024 canvas because that fed a fixed-size Mask R-CNN. That model
    is gone, and Model 2 (pattern extraction) isn't specified yet, so
    extraction stays a tight, native-resolution/native-aspect-ratio crop -
    forcing a canonical size now would bake in an assumption we might have
    to undo later.

    Attributes:
        background_fill: RGB color used to replace every pixel outside the
            segmentation mask.
        crop_margin_px: Extra pixels of padding added around the mask's
            bounding box before cropping, so the fish silhouette isn't cut
            off flush at the edge.
    """

    background_fill: tuple[int, int, int] = (127, 127, 127)
    crop_margin_px: int = 16


@dataclass
class PipelineConfig:
    """Top-level configuration for the fish-extraction pipeline.

    Attributes:
        raw_images_root: Root directory holding one subfolder per genus,
            each containing one zipped-or-unzipped species folder (shared
            with ``dataset_builder``).
        extracted_root: Root directory where accepted extractions are
            written, mirroring the genus/species layout of raw_images_root.
        state_path: Where resumable per-image progress is persisted.
        metadata_csv_path: Where per-image detection/QA metadata rows are
            appended.
        review_path: Where the human-review HTML page is written.
        detection: Nested detection settings.
        segmentation: Nested segmentation settings.
        qa_gate: Nested auto-accept/flag threshold settings.
        extraction: Nested output-image settings.
    """

    raw_images_root: Path = field(default_factory=lambda: Path("data/raw_images"))
    extracted_root: Path = field(default_factory=lambda: Path("data/extracted_fish"))
    state_path: Path = field(
        default_factory=lambda: Path("data/fish_extraction_state.json")
    )
    metadata_csv_path: Path = field(
        default_factory=lambda: Path("reports/fish_extraction_log.csv")
    )
    review_path: Path = field(
        default_factory=lambda: Path("reports/fish_extraction_review.html")
    )
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    qa_gate: QAGateConfig = field(default_factory=QAGateConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
