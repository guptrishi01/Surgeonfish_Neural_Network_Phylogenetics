"""Unit tests for pipeline orchestration, with detector/segmenter mocked out.

None of these tests load a real model - GroundingDinoDetector/Sam2Segmenter
are replaced with stubs, confirming the wiring (QA gate -> extraction ->
state -> metadata log) without needing a GPU or the optional `vision`
dependencies installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fish_extractor.config import PipelineConfig
from fish_extractor.pipeline import FishExtractorPipeline
from fish_extractor.qa_gate import Box
from fish_extractor.state import ExtractionState


class _StubDetector:
    """Returns a fixed list of boxes for every image, records call count."""

    def __init__(self, boxes: list[Box]) -> None:
        self.boxes = boxes
        self.call_count = 0

    def detect(self, image: Image.Image) -> list[Box]:
        self.call_count += 1
        return self.boxes


class _StubSegmenter:
    """Returns a mask covering the box region, records call count."""

    def __init__(self) -> None:
        self.call_count = 0

    def segment(self, image: Image.Image, box: Box) -> np.ndarray:
        self.call_count += 1
        width, height = image.size
        mask = np.zeros((height, width), dtype=bool)
        mask[int(box.y0):int(box.y1), int(box.x0):int(box.x1)] = True
        return mask


class _RaisingDetector:
    """Fails the test loudly if detect() is ever called (proves skip-on-resume)."""

    def detect(self, image: Image.Image) -> list[Box]:
        raise AssertionError("detector should not be called for an already-processed image")


def _make_species_image(
    root: Path, genus: str, species: str, filename: str, size=(400, 300)
) -> Path:
    species_dir = root / genus / species
    species_dir.mkdir(parents=True, exist_ok=True)
    path = species_dir / filename
    Image.new("RGB", size, (50, 80, 120)).save(path)
    return path


def _centered_box(size=(400, 300), confidence=0.9) -> Box:
    width, height = size
    return Box(
        x0=width * 0.25, y0=height * 0.25, x1=width * 0.75, y1=height * 0.75,
        confidence=confidence,
    )


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        raw_images_root=tmp_path / "raw_images",
        extracted_root=tmp_path / "extracted_fish",
        state_path=tmp_path / "state.json",
        metadata_csv_path=tmp_path / "reports" / "log.csv",
        review_path=tmp_path / "reports" / "review.html",
    )


def test_accepted_image_writes_extracted_image_and_mask(tmp_path: Path):
    config = _config(tmp_path)
    _make_species_image(
        config.raw_images_root, "Acanthurus", "Acanthurus_guttatus", "000_reference.jpg"
    )
    detector = _StubDetector([_centered_box()])
    segmenter = _StubSegmenter()
    pipeline = FishExtractorPipeline(config, detector=detector, segmenter=segmenter)

    results = pipeline.run()

    assert len(results) == 1
    assert results[0].status == "accepted"
    extracted_dir = config.extracted_root / "Acanthurus" / "Acanthurus_guttatus"
    assert (extracted_dir / "000_reference.png").exists()
    assert (extracted_dir / "000_reference_mask.png").exists()
    assert config.metadata_csv_path.exists()


def test_flagged_image_does_not_call_segmenter_or_write_extraction(tmp_path: Path):
    config = _config(tmp_path)
    _make_species_image(config.raw_images_root, "Naso", "Naso_annulatus", "001_gbif_1.jpg")
    detector = _StubDetector([])  # no boxes -> no_detection
    segmenter = _StubSegmenter()
    pipeline = FishExtractorPipeline(config, detector=detector, segmenter=segmenter)

    results = pipeline.run()

    assert results[0].status == "flagged"
    assert results[0].reason == "no_detection"
    assert segmenter.call_count == 0
    assert not (config.extracted_root / "Naso" / "Naso_annulatus").exists()


def test_multiple_boxes_is_flagged_multiple_fish(tmp_path: Path):
    config = _config(tmp_path)
    _make_species_image(
        config.raw_images_root, "Zebrasoma", "Zebrasoma_flavescens", "002_gbif_2.jpg"
    )
    # Genuinely separate (non-overlapping) boxes - two identically-positioned
    # boxes would now be deduplicated as one detection, see qa_gate.py.
    boxes = [
        Box(x0=0, y0=0, x1=100, y1=100, confidence=0.9),
        Box(x0=300, y0=200, x1=400, y1=300, confidence=0.8),
    ]
    pipeline = FishExtractorPipeline(
        config, detector=_StubDetector(boxes), segmenter=_StubSegmenter()
    )

    results = pipeline.run()

    assert results[0].status == "flagged"
    assert results[0].reason == "multiple_fish"


def test_rerun_skips_already_processed_images(tmp_path: Path):
    config = _config(tmp_path)
    _make_species_image(
        config.raw_images_root, "Acanthurus", "Acanthurus_guttatus", "000_reference.jpg"
    )
    first_pipeline = FishExtractorPipeline(
        config, detector=_StubDetector([_centered_box()]), segmenter=_StubSegmenter()
    )
    first_results = first_pipeline.run()
    assert len(first_results) == 1

    second_pipeline = FishExtractorPipeline(
        config, detector=_RaisingDetector(), segmenter=_StubSegmenter()
    )
    second_results = second_pipeline.run()

    assert second_results == []


def test_species_filter_restricts_which_images_are_processed(tmp_path: Path):
    config = _config(tmp_path)
    _make_species_image(
        config.raw_images_root, "Acanthurus", "Acanthurus_guttatus", "000_reference.jpg"
    )
    _make_species_image(
        config.raw_images_root, "Zebrasoma", "Zebrasoma_flavescens", "000_reference.jpg"
    )
    pipeline = FishExtractorPipeline(
        config, detector=_StubDetector([_centered_box()]), segmenter=_StubSegmenter()
    )

    results = pipeline.run(species_filter={"Acanthurus guttatus"})

    assert len(results) == 1
    assert results[0].image_key.startswith("Acanthurus/")


def test_unreadable_image_is_flagged_not_raised(tmp_path: Path):
    config = _config(tmp_path)
    species_dir = config.raw_images_root / "Acanthurus" / "Acanthurus_guttatus"
    species_dir.mkdir(parents=True)
    corrupt_path = species_dir / "000_reference.jpg"
    corrupt_path.write_bytes(b"not a real image")
    pipeline = FishExtractorPipeline(
        config, detector=_StubDetector([_centered_box()]), segmenter=_StubSegmenter()
    )

    results = pipeline.run()

    assert results[0].status == "flagged"
    assert results[0].reason == "unreadable"


def test_extraction_replaces_background_and_crops_to_mask(tmp_path: Path):
    config = _config(tmp_path)
    _make_species_image(
        config.raw_images_root, "Acanthurus", "Acanthurus_guttatus", "000_reference.jpg"
    )
    pipeline = FishExtractorPipeline(
        config, detector=_StubDetector([_centered_box()]), segmenter=_StubSegmenter()
    )

    pipeline.run()

    extracted_path = (
        config.extracted_root / "Acanthurus" / "Acanthurus_guttatus" / "000_reference.png"
    )
    extracted = Image.open(extracted_path)
    # Cropped to the box (200x150) plus 2*margin, not the full 400x300 source.
    assert extracted.size[0] < 400
    assert extracted.size[1] < 300


def _off_center_box(size=(400, 300), confidence=0.9) -> Box:
    """A single, real box that fails only the off_center threshold (area is fine)."""
    return Box(x0=0, y0=0, x1=100, y1=75, confidence=confidence)


def test_force_accept_flagged_extracts_using_the_stored_box(tmp_path: Path):
    config = _config(tmp_path)
    _make_species_image(
        config.raw_images_root, "Acanthurus", "Acanthurus_guttatus", "001_gbif_1.jpg"
    )
    segmenter = _StubSegmenter()
    pipeline = FishExtractorPipeline(
        config, detector=_StubDetector([_off_center_box()]), segmenter=segmenter
    )
    flagged = pipeline.run()
    assert flagged[0].status == "flagged"
    assert flagged[0].reason == "off_center"
    assert segmenter.call_count == 0  # never segmented while flagged

    results = pipeline.force_accept_flagged()

    assert len(results) == 1
    assert results[0].status == "accepted"
    assert results[0].reason == "human_confirmed"
    assert segmenter.call_count == 1
    extracted_dir = config.extracted_root / "Acanthurus" / "Acanthurus_guttatus"
    assert (extracted_dir / "001_gbif_1.png").exists()
    assert (extracted_dir / "001_gbif_1_mask.png").exists()


def test_force_accept_flagged_skips_entries_with_no_stored_box(tmp_path: Path):
    config = _config(tmp_path)
    _make_species_image(config.raw_images_root, "Naso", "Naso_annulatus", "001_gbif_1.jpg")
    pipeline = FishExtractorPipeline(
        config, detector=_StubDetector([]), segmenter=_StubSegmenter()  # no_detection, box=None
    )
    flagged = pipeline.run()
    assert flagged[0].reason == "no_detection"

    results = pipeline.force_accept_flagged()

    assert results == []
    state = ExtractionState(config.state_path)
    assert state.get("Naso/Naso_annulatus/001_gbif_1.jpg").status == "flagged"


def test_force_accept_flagged_respects_image_keys_filter(tmp_path: Path):
    config = _config(tmp_path)
    _make_species_image(
        config.raw_images_root, "Acanthurus", "Acanthurus_guttatus", "001_gbif_1.jpg"
    )
    _make_species_image(
        config.raw_images_root, "Zebrasoma", "Zebrasoma_flavescens", "001_gbif_1.jpg"
    )
    pipeline = FishExtractorPipeline(
        config, detector=_StubDetector([_off_center_box()]), segmenter=_StubSegmenter()
    )
    pipeline.run()

    results = pipeline.force_accept_flagged(
        image_keys={"Acanthurus/Acanthurus_guttatus/001_gbif_1.jpg"}
    )

    assert len(results) == 1
    assert results[0].image_key == "Acanthurus/Acanthurus_guttatus/001_gbif_1.jpg"
    state = ExtractionState(config.state_path)
    assert state.get("Zebrasoma/Zebrasoma_flavescens/001_gbif_1.jpg").status == "flagged"


def test_force_accept_with_top_box_extracts_using_the_highest_confidence_box(tmp_path: Path):
    config = _config(tmp_path)
    _make_species_image(
        config.raw_images_root, "Acanthurus", "Acanthurus_guttatus", "001_gbif_1.jpg"
    )
    # Two genuinely separate boxes - flags as multiple_fish, no box stored.
    boxes = [
        Box(x0=10, y0=10, x1=110, y1=110, confidence=0.6),
        Box(x0=200, y0=150, x1=340, y1=260, confidence=0.9),  # higher confidence
    ]
    detector = _StubDetector(boxes)
    segmenter = _StubSegmenter()
    pipeline = FishExtractorPipeline(config, detector=detector, segmenter=segmenter)
    flagged = pipeline.run()
    assert flagged[0].reason == "multiple_fish"
    assert segmenter.call_count == 0

    results = pipeline.force_accept_with_top_box(
        image_keys={"Acanthurus/Acanthurus_guttatus/001_gbif_1.jpg"}
    )

    assert len(results) == 1
    assert results[0].status == "accepted"
    assert results[0].reason == "human_confirmed_multi_box"
    assert segmenter.call_count == 1
    state = ExtractionState(config.state_path)
    entry = state.get("Acanthurus/Acanthurus_guttatus/001_gbif_1.jpg")
    assert entry.status == "accepted"
    assert entry.box[4] == 0.9  # the higher-confidence box was used
    extracted_dir = config.extracted_root / "Acanthurus" / "Acanthurus_guttatus"
    assert (extracted_dir / "001_gbif_1.png").exists()


def test_force_accept_with_top_box_skips_and_reports_when_nothing_qualifies(tmp_path: Path):
    config = _config(tmp_path)
    _make_species_image(config.raw_images_root, "Naso", "Naso_annulatus", "001_gbif_1.jpg")
    pipeline = FishExtractorPipeline(
        config, detector=_StubDetector([]), segmenter=_StubSegmenter()  # no_detection
    )
    pipeline.run()

    results = pipeline.force_accept_with_top_box(
        image_keys={"Naso/Naso_annulatus/001_gbif_1.jpg"}
    )

    assert results == []
    state = ExtractionState(config.state_path)
    assert state.get("Naso/Naso_annulatus/001_gbif_1.jpg").status == "flagged"


def test_saved_mask_is_cropped_to_match_the_extracted_image(tmp_path: Path):
    config = _config(tmp_path)
    _make_species_image(
        config.raw_images_root, "Acanthurus", "Acanthurus_guttatus", "000_reference.jpg"
    )
    pipeline = FishExtractorPipeline(
        config, detector=_StubDetector([_centered_box()]), segmenter=_StubSegmenter()
    )

    pipeline.run()

    species_dir = config.extracted_root / "Acanthurus" / "Acanthurus_guttatus"
    extracted = Image.open(species_dir / "000_reference.png")
    mask = Image.open(species_dir / "000_reference_mask.png")

    assert mask.size == extracted.size
    # A corner pixel (outside the mask) should be background_fill, not the
    # original 400x300 source image's fill color (50, 80, 120).
    assert extracted.getpixel((0, 0)) != (50, 80, 120)
