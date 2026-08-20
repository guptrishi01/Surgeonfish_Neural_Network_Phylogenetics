"""Unit tests for the --run gate in the fish_extractor CLI entry point.

Default (no --run) must never load a detection/segmentation model - this
lets a future end-to-end pipeline invocation, or a plain accidental run,
skip the multi-GB model download/load entirely unless explicitly asked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from fish_extractor import cli
from fish_extractor.config import PipelineConfig


class _RecordingPipeline:
    """Stands in for FishExtractorPipeline; records whether .run() fired."""

    instances: list["_RecordingPipeline"] = []

    def __init__(self, config):
        self.config = config
        self.ran = False
        _RecordingPipeline.instances.append(self)

    def run(self, species_filter=None):
        self.ran = True
        return []


@pytest.fixture(autouse=True)
def _reset_recording_pipeline():
    _RecordingPipeline.instances.clear()
    yield
    _RecordingPipeline.instances.clear()


@pytest.fixture
def _patched_cli(tmp_path: Path, monkeypatch):
    config = PipelineConfig(
        raw_images_root=tmp_path / "raw_images",
        extracted_root=tmp_path / "extracted_fish",
        state_path=tmp_path / "state.json",
        metadata_csv_path=tmp_path / "reports" / "log.csv",
        review_path=tmp_path / "reports" / "review.html",
    )
    monkeypatch.setattr(cli, "PipelineConfig", lambda: config)
    monkeypatch.setattr(cli, "FishExtractorPipeline", _RecordingPipeline)

    def _fake_generate_review_html(cfg, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(cli, "generate_review_html", _fake_generate_review_html)
    return config


def test_default_run_never_loads_a_model(monkeypatch, _patched_cli):
    monkeypatch.setattr(sys, "argv", ["cli.py"])

    cli.main()

    assert _RecordingPipeline.instances == []


def test_run_flag_runs_the_pipeline(monkeypatch, _patched_cli):
    monkeypatch.setattr(sys, "argv", ["cli.py", "--run"])

    cli.main()

    assert len(_RecordingPipeline.instances) == 1
    assert _RecordingPipeline.instances[0].ran


def test_apply_review_without_run_applies_but_does_not_extract(
    monkeypatch, _patched_cli, tmp_path: Path
):
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text('{"excluded": []}', encoding="utf-8")

    def _fake_apply_review(cfg, path):
        return 0

    monkeypatch.setattr(cli, "apply_review_feedback", _fake_apply_review)
    monkeypatch.setattr(sys, "argv", ["cli.py", "--apply-review", str(feedback_path)])

    cli.main()

    assert _RecordingPipeline.instances == []
