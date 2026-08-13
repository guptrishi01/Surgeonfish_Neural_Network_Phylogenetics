"""Unit tests for the human review page and feedback-driven backfill."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from dataset_builder.config import PipelineConfig
from dataset_builder.review import apply_review_feedback, generate_review_html
from dataset_builder.state import PipelineState


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), (10, 20, 30)).save(path)


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        raw_images_root=tmp_path / "raw_images",
        target_per_species=3,
        state_path=tmp_path / "state.json",
        metadata_csv_path=tmp_path / "sourcing_log.csv",
    )


def _seed_species(config: PipelineConfig) -> Path:
    species_dir = config.raw_images_root / "Zebrasoma" / "Zebrasoma_flavescens"
    _make_image(species_dir / "000_reference.jpg")
    _make_image(species_dir / "001_gbif_111.jpg")
    _make_image(species_dir / "002_gbif_222.jpg")
    return species_dir


def test_generate_review_html_lists_species_and_images(tmp_path: Path):
    config = _config(tmp_path)
    _seed_species(config)

    output = generate_review_html(config, tmp_path / "reports" / "review.html")
    html = output.read_text(encoding="utf-8")

    assert "Zebrasoma flavescens" in html
    assert "001_gbif_111.jpg" in html
    assert "002_gbif_222.jpg" in html
    assert 'class="image-card reference"' in html


def test_reference_image_has_no_keep_checkbox(tmp_path: Path):
    config = _config(tmp_path)
    _seed_species(config)

    output = generate_review_html(config, tmp_path / "reports" / "review.html")
    html = output.read_text(encoding="utf-8")

    reference_block = html.split('class="image-card reference"')[1].split("</div>")[0]
    assert "checkbox" not in reference_block


def test_apply_review_feedback_removes_rejected_image_and_updates_state(tmp_path: Path):
    config = _config(tmp_path)
    species_dir = _seed_species(config)

    # Seed state as if a prior collection run had already accepted these 3.
    state = PipelineState(config.state_path)
    species_state = state.get("Zebrasoma flavescens")
    species_state.accepted_count = 3
    species_state.exhausted = True
    state.save()

    feedback_path = tmp_path / "review_feedback.json"
    feedback_path.write_text(json.dumps({"Zebrasoma flavescens": ["001_gbif_111.jpg"]}))

    removed = apply_review_feedback(config, feedback_path)

    assert removed == {"Zebrasoma flavescens": 1}
    assert not (species_dir / "001_gbif_111.jpg").exists()
    assert (species_dir / "002_gbif_222.jpg").exists()
    assert (species_dir / "000_reference.jpg").exists()

    reloaded = PipelineState(config.state_path).get("Zebrasoma flavescens")
    assert reloaded.accepted_count == 2
    assert 111 in reloaded.seen_occurrence_keys
    assert reloaded.exhausted is False


def test_apply_review_feedback_refuses_to_delete_reference(tmp_path: Path):
    config = _config(tmp_path)
    species_dir = _seed_species(config)

    feedback_path = tmp_path / "review_feedback.json"
    feedback_path.write_text(json.dumps({"Zebrasoma flavescens": ["000_reference.jpg"]}))

    removed = apply_review_feedback(config, feedback_path)

    assert removed == {}
    assert (species_dir / "000_reference.jpg").exists()
