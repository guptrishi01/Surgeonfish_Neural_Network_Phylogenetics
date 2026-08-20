"""Unit tests for pattern-extraction pipeline orchestration."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from pattern_extractor.config import PipelineConfig
from pattern_extractor.pipeline import PatternExtractorPipeline


def _save_pair(species_dir: Path, stem: str, size=(30, 30), color=(80, 120, 60)) -> None:
    species_dir.mkdir(parents=True, exist_ok=True)
    height, width = size
    image = np.full((height, width, 3), color, dtype=np.uint8)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    Image.fromarray(image).save(species_dir / f"{stem}.png")
    Image.fromarray(mask).save(species_dir / f"{stem}_mask.png")


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        extracted_root=tmp_path / "extracted_fish",
        output_csv_path=tmp_path / "reports" / "pattern_features.csv",
    )


def test_run_writes_one_row_per_image_across_species(tmp_path: Path):
    config = _config(tmp_path)
    species_dir = config.extracted_root / "Acanthurus" / "Acanthurus_guttatus"
    _save_pair(species_dir, "000_reference")
    _save_pair(species_dir, "001_gbif_1")
    other_dir = config.extracted_root / "Zebrasoma" / "Zebrasoma_flavescens"
    _save_pair(other_dir, "000_reference")

    pipeline = PatternExtractorPipeline(config)
    rows = pipeline.run()

    assert len(rows) == 3
    assert config.output_csv_path.exists()


def test_csv_has_expected_header_and_row_count(tmp_path: Path):
    config = _config(tmp_path)
    species_dir = config.extracted_root / "Acanthurus" / "Acanthurus_guttatus"
    _save_pair(species_dir, "000_reference")

    PatternExtractorPipeline(config).run()

    with open(config.output_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert "is_solid" in reader.fieldnames
    assert "spot_present" in reader.fieldnames
    assert "stripe_present" in reader.fieldnames


def test_species_without_a_reference_stem_falls_back_to_first_image(tmp_path: Path):
    config = _config(tmp_path)
    species_dir = config.extracted_root / "Naso" / "Naso_annulatus"
    _save_pair(species_dir, "001_gbif_1")  # no "000_reference" present

    rows = PatternExtractorPipeline(config).run()

    assert len(rows) == 1


def test_image_missing_its_mask_file_is_skipped_but_others_still_processed(tmp_path: Path):
    config = _config(tmp_path)
    species_dir = config.extracted_root / "Acanthurus" / "Acanthurus_guttatus"
    _save_pair(species_dir, "000_reference")
    _save_pair(species_dir, "001_gbif_1")
    (species_dir / "001_gbif_1_mask.png").unlink()  # simulate a missing mask

    rows = PatternExtractorPipeline(config).run()

    assert len(rows) == 1
    assert rows[0]["image_key"] == "Acanthurus/Acanthurus_guttatus/000_reference"


def test_species_filter_restricts_which_species_are_processed(tmp_path: Path):
    config = _config(tmp_path)
    _save_pair(config.extracted_root / "Acanthurus" / "Acanthurus_guttatus", "000_reference")
    _save_pair(config.extracted_root / "Zebrasoma" / "Zebrasoma_flavescens", "000_reference")

    rows = PatternExtractorPipeline(config).run(species_filter={"Acanthurus guttatus"})

    assert len(rows) == 1
    assert rows[0]["image_key"].startswith("Acanthurus/")


def test_image_key_matches_relative_path_and_stem(tmp_path: Path):
    config = _config(tmp_path)
    _save_pair(config.extracted_root / "Acanthurus" / "Acanthurus_guttatus", "000_reference")

    rows = PatternExtractorPipeline(config).run()

    assert rows[0]["image_key"] == "Acanthurus/Acanthurus_guttatus/000_reference"
