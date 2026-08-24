"""Unit tests for the fish-extraction human review page and feedback loop."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from fish_extractor.config import PipelineConfig
from fish_extractor.review import (
    apply_review_feedback,
    generate_review_html,
    reset_flagged_for_reprocessing,
)
from fish_extractor.state import ExtractionState


def _make_image(path: Path, size=(100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path)


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        raw_images_root=tmp_path / "raw_images",
        extracted_root=tmp_path / "extracted_fish",
        state_path=tmp_path / "state.json",
        metadata_csv_path=tmp_path / "reports" / "log.csv",
        review_path=tmp_path / "reports" / "review.html",
    )


def test_generate_review_html_lists_only_flagged_images(tmp_path: Path):
    config = _config(tmp_path)
    _make_image(config.raw_images_root / "Naso" / "Naso_annulatus" / "001_gbif_1.jpg")
    _make_image(config.raw_images_root / "Naso" / "Naso_annulatus" / "002_gbif_2.jpg")

    state = ExtractionState(config.state_path)
    flagged = state.get("Naso/Naso_annulatus/001_gbif_1.jpg")
    flagged.status, flagged.reason = "flagged", "multiple_fish"
    accepted = state.get("Naso/Naso_annulatus/002_gbif_2.jpg")
    accepted.status, accepted.reason = "accepted", "ok"
    state.save()

    output = generate_review_html(config, tmp_path / "reports" / "review.html")
    html = output.read_text(encoding="utf-8")

    assert "001_gbif_1.jpg" in html
    assert "002_gbif_2.jpg" not in html
    assert "multiple_fish" in html


def test_flagged_image_with_box_gets_overlay_style(tmp_path: Path):
    config = _config(tmp_path)
    image_path = config.raw_images_root / "Naso" / "Naso_annulatus" / "001_gbif_1.jpg"
    _make_image(image_path, size=(100, 80))

    state = ExtractionState(config.state_path)
    entry = state.get("Naso/Naso_annulatus/001_gbif_1.jpg")
    entry.status, entry.reason = "flagged", "off_center"
    entry.box = [0.0, 0.0, 20.0, 16.0, 0.9]  # 20% width, 20% height, top-left
    state.save()

    output = generate_review_html(config, tmp_path / "reports" / "review.html")
    html = output.read_text(encoding="utf-8")

    assert "box-overlay" in html
    assert "width:20.00%" in html


def test_apply_review_feedback_marks_excluded_and_ignores_non_flagged(tmp_path: Path):
    config = _config(tmp_path)
    state = ExtractionState(config.state_path)
    flagged = state.get("Naso/Naso_annulatus/001_gbif_1.jpg")
    flagged.status = "flagged"
    already_accepted = state.get("Naso/Naso_annulatus/002_gbif_2.jpg")
    already_accepted.status = "accepted"
    state.save()

    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(
        json.dumps({"excluded": [
            "Naso/Naso_annulatus/001_gbif_1.jpg",
            "Naso/Naso_annulatus/002_gbif_2.jpg",
        ]}),
        encoding="utf-8",
    )

    count = apply_review_feedback(config, feedback_path)

    assert count == 1  # the already-accepted one is skipped, not force-excluded
    reloaded = ExtractionState(config.state_path)
    assert reloaded.get("Naso/Naso_annulatus/001_gbif_1.jpg").status == "excluded"
    assert reloaded.get("Naso/Naso_annulatus/002_gbif_2.jpg").status == "accepted"


def test_reset_flagged_for_reprocessing_clears_flagged_entries_only(tmp_path: Path):
    config = _config(tmp_path)
    state = ExtractionState(config.state_path)
    flagged = state.get("Naso/Naso_annulatus/001_gbif_1.jpg")
    flagged.status, flagged.reason = "flagged", "multiple_fish"
    flagged.box = [0.0, 0.0, 20.0, 16.0, 0.9]
    already_accepted = state.get("Naso/Naso_annulatus/002_gbif_2.jpg")
    already_accepted.status, already_accepted.reason = "accepted", "ok"
    state.save()

    count = reset_flagged_for_reprocessing(config)

    assert count == 1
    reloaded = ExtractionState(config.state_path)
    reset_entry = reloaded.get("Naso/Naso_annulatus/001_gbif_1.jpg")
    assert reset_entry.status == "pending"
    assert reset_entry.reason == ""
    assert reset_entry.box is None
    assert reloaded.get("Naso/Naso_annulatus/002_gbif_2.jpg").status == "accepted"


def test_reset_flagged_for_reprocessing_respects_image_keys_filter(tmp_path: Path):
    config = _config(tmp_path)
    state = ExtractionState(config.state_path)
    state.get("Naso/Naso_annulatus/001_gbif_1.jpg").status = "flagged"
    state.get("Naso/Naso_annulatus/002_gbif_2.jpg").status = "flagged"
    state.save()

    count = reset_flagged_for_reprocessing(
        config, image_keys={"Naso/Naso_annulatus/001_gbif_1.jpg"}
    )

    assert count == 1
    reloaded = ExtractionState(config.state_path)
    assert reloaded.get("Naso/Naso_annulatus/001_gbif_1.jpg").status == "pending"
    assert reloaded.get("Naso/Naso_annulatus/002_gbif_2.jpg").status == "flagged"
