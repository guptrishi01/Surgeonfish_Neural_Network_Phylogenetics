"""Unit tests for the manual pattern-validation split."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pattern_extractor.config import PipelineConfig, ValidationConfig
from pattern_extractor.pipeline import PatternExtractorPipeline
from pattern_extractor.validation import (
    compare_to_features,
    generate_labeling_html,
    load_labels,
    select_sample,
    summarize_agreement,
)


def _save_pair(species_dir: Path, stem: str, size=(30, 30), color=(80, 120, 60)) -> None:
    species_dir.mkdir(parents=True, exist_ok=True)
    height, width = size
    image = np.full((height, width, 3), color, dtype=np.uint8)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    Image.fromarray(image).save(species_dir / f"{stem}.png")
    Image.fromarray(mask).save(species_dir / f"{stem}_mask.png")


def _pipeline_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        extracted_root=tmp_path / "extracted_fish",
        output_csv_path=tmp_path / "reports" / "pattern_features.csv",
    )


def _validation_config(tmp_path: Path, **overrides) -> ValidationConfig:
    defaults = {
        "labeling_html_path": tmp_path / "reports" / "labeling.html",
        "labels_json_path": tmp_path / "reports" / "labels.json",
        "report_csv_path": tmp_path / "reports" / "report.csv",
    }
    defaults.update(overrides)
    return ValidationConfig(**defaults)


def test_select_sample_covers_every_species_and_excludes_reference(tmp_path: Path):
    pipeline_config = _pipeline_config(tmp_path)
    species_a = pipeline_config.extracted_root / "Acanthurus" / "Acanthurus_guttatus"
    _save_pair(species_a, "000_reference")
    for i in range(5):
        _save_pair(species_a, f"{i:03d}_gbif_{i}", color=(80 + i, 120, 60))
    species_b = pipeline_config.extracted_root / "Zebrasoma" / "Zebrasoma_flavescens"
    _save_pair(species_b, "000_reference")
    _save_pair(species_b, "001_gbif_1", color=(200, 200, 30))

    PatternExtractorPipeline(pipeline_config).run()
    validation_config = _validation_config(tmp_path, min_per_species=1, sample_fraction=0.2)

    sample = select_sample(pipeline_config, validation_config)

    assert "Acanthurus/Acanthurus_guttatus/000_reference" not in sample
    assert "Zebrasoma/Zebrasoma_flavescens/000_reference" not in sample
    species_in_sample = {key.split("/")[1] for key in sample}
    assert species_in_sample == {"Acanthurus_guttatus", "Zebrasoma_flavescens"}


def test_select_sample_is_deterministic_given_same_seed(tmp_path: Path):
    pipeline_config = _pipeline_config(tmp_path)
    species_dir = pipeline_config.extracted_root / "Acanthurus" / "Acanthurus_guttatus"
    _save_pair(species_dir, "000_reference")
    for i in range(10):
        _save_pair(species_dir, f"{i:03d}_gbif_{i}", color=(80 + i, 120, 60))
    PatternExtractorPipeline(pipeline_config).run()
    validation_config = _validation_config(tmp_path, random_seed=42)

    first = select_sample(pipeline_config, validation_config)
    second = select_sample(pipeline_config, validation_config)

    assert first == second


def test_select_sample_respects_min_per_species_for_a_sparse_species(tmp_path: Path):
    pipeline_config = _pipeline_config(tmp_path)
    species_dir = pipeline_config.extracted_root / "Naso" / "Naso_maculatus"
    _save_pair(species_dir, "000_reference")
    _save_pair(species_dir, "001_gbif_1")  # only 1 non-reference image
    PatternExtractorPipeline(pipeline_config).run()
    validation_config = _validation_config(tmp_path, sample_fraction=0.2, min_per_species=1)

    sample = select_sample(pipeline_config, validation_config)

    assert sample == ["Naso/Naso_maculatus/001_gbif_1"]


def test_generate_labeling_html_embeds_sampled_images(tmp_path: Path):
    pipeline_config = _pipeline_config(tmp_path)
    species_dir = pipeline_config.extracted_root / "Acanthurus" / "Acanthurus_guttatus"
    _save_pair(species_dir, "001_gbif_1")
    validation_config = _validation_config(tmp_path)

    output_path = generate_labeling_html(
        pipeline_config, validation_config, ["Acanthurus/Acanthurus_guttatus/001_gbif_1"]
    )

    html = output_path.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in html
    assert 'data-key="Acanthurus/Acanthurus_guttatus/001_gbif_1"' in html


def test_generate_labeling_html_skips_missing_image_file(tmp_path: Path):
    pipeline_config = _pipeline_config(tmp_path)
    validation_config = _validation_config(tmp_path)

    output_path = generate_labeling_html(
        pipeline_config, validation_config, ["Acanthurus/Acanthurus_guttatus/does_not_exist"]
    )

    html = output_path.read_text(encoding="utf-8")
    assert "does_not_exist" not in html
    assert "0 image(s)" in html


def test_load_labels_reads_exported_json(tmp_path: Path):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "labels": {
                    "A/a/1": {
                        "is_solid": True,
                        "stripe_present": False,
                        "spot_present": False,
                        "color_count": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    labels = load_labels(labels_path)

    assert labels["A/a/1"]["is_solid"] is True


def _write_feature_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_key", "is_reference", "is_solid", "stripe_present", "spot_present"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_labels(path: Path, labels: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"labels": labels}), encoding="utf-8")


def test_compare_to_features_flags_agreement_and_disagreement(tmp_path: Path):
    pipeline_config = _pipeline_config(tmp_path)
    _write_feature_csv(
        pipeline_config.output_csv_path,
        [
            {
                "image_key": "A/a/1",
                "is_reference": "False",
                "is_solid": "True",
                "stripe_present": "False",
                "spot_present": "False",
            },
            {
                "image_key": "A/a/2",
                "is_reference": "False",
                "is_solid": "False",
                "stripe_present": "True",
                "spot_present": "False",
            },
        ],
    )
    validation_config = _validation_config(tmp_path)
    _write_labels(
        validation_config.labels_json_path,
        {
            # agrees on every dimension
            "A/a/1": {"is_solid": True, "stripe_present": False, "spot_present": False},
            # disagrees on is_solid and stripe_present
            "A/a/2": {"is_solid": True, "stripe_present": False, "spot_present": False},
        },
    )

    rows = compare_to_features(pipeline_config, validation_config)

    by_key_dim = {(r.image_key, r.dimension): r.agree for r in rows}
    assert by_key_dim[("A/a/1", "is_solid")] is True
    assert by_key_dim[("A/a/2", "is_solid")] is False
    assert by_key_dim[("A/a/2", "stripe_present")] is False
    assert by_key_dim[("A/a/2", "spot_present")] is True
    assert validation_config.report_csv_path.exists()


def test_compare_to_features_skips_labels_with_no_matching_feature_row(tmp_path: Path):
    pipeline_config = _pipeline_config(tmp_path)
    _write_feature_csv(
        pipeline_config.output_csv_path,
        [
            {
                "image_key": "A/a/1",
                "is_reference": "False",
                "is_solid": "True",
                "stripe_present": "False",
                "spot_present": "False",
            }
        ],
    )
    validation_config = _validation_config(tmp_path)
    _write_labels(
        validation_config.labels_json_path,
        {
            "A/a/1": {"is_solid": True, "stripe_present": False, "spot_present": False},
            "A/a/missing": {"is_solid": True, "stripe_present": False, "spot_present": False},
        },
    )

    rows = compare_to_features(pipeline_config, validation_config)

    assert {r.image_key for r in rows} == {"A/a/1"}


def test_summarize_agreement_computes_per_dimension_rate(tmp_path: Path):
    pipeline_config = _pipeline_config(tmp_path)
    _write_feature_csv(
        pipeline_config.output_csv_path,
        [
            {
                "image_key": "A/a/1",
                "is_reference": "False",
                "is_solid": "True",
                "stripe_present": "False",
                "spot_present": "False",
            },
            {
                "image_key": "A/a/2",
                "is_reference": "False",
                "is_solid": "False",
                "stripe_present": "False",
                "spot_present": "False",
            },
        ],
    )
    validation_config = _validation_config(tmp_path)
    _write_labels(
        validation_config.labels_json_path,
        {
            # all agree
            "A/a/1": {"is_solid": True, "stripe_present": False, "spot_present": False},
            # is_solid disagrees
            "A/a/2": {"is_solid": True, "stripe_present": False, "spot_present": False},
        },
    )

    rows = compare_to_features(pipeline_config, validation_config)
    summary = summarize_agreement(rows)

    assert summary["is_solid"] == 0.5
    assert summary["stripe_present"] == 1.0
    assert summary["spot_present"] == 1.0
