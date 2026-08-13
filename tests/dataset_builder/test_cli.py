"""Unit tests for the --scrape-web gate in the CLI entry point.

Default (no --scrape-web) must never touch the network - this is what
lets a future end-to-end pipeline invocation, or a plain accidental run,
skip GBIF entirely unless explicitly asked for it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dataset_builder import cli
from dataset_builder.config import PipelineConfig


class _RecordingPipeline:
    """Stands in for DatasetBuilderPipeline; records whether .run() fired."""

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
        state_path=tmp_path / "state.json",
        metadata_csv_path=tmp_path / "sourcing_log.csv",
    )
    config.raw_images_root.mkdir(parents=True)
    monkeypatch.setattr(cli, "PipelineConfig", lambda: config)
    monkeypatch.setattr(cli, "DatasetBuilderPipeline", _RecordingPipeline)
    def _fake_generate_review_html(cfg, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(cli, "generate_review_html", _fake_generate_review_html)
    return config


def test_default_run_never_touches_the_network(monkeypatch, _patched_cli, tmp_path: Path):
    monkeypatch.setattr(sys, "argv", ["cli.py"])

    cli.main()

    assert _RecordingPipeline.instances == []  # never even constructed


def test_scrape_web_flag_runs_the_pipeline(monkeypatch, _patched_cli, tmp_path: Path):
    monkeypatch.setattr(sys, "argv", ["cli.py", "--scrape-web"])

    cli.main()

    assert len(_RecordingPipeline.instances) == 1
    assert _RecordingPipeline.instances[0].ran


def test_apply_review_without_scrape_web_purges_but_does_not_backfill(
    monkeypatch, _patched_cli, tmp_path: Path
):
    feedback_path = tmp_path / "review_feedback.json"
    feedback_path.write_text("{}", encoding="utf-8")
    called = {"ran": False}

    def _fake_apply_review(cfg, path):
        called["ran"] = True
        return {}

    monkeypatch.setattr(cli, "apply_review_feedback", _fake_apply_review)
    monkeypatch.setattr(sys, "argv", ["cli.py", "--apply-review", str(feedback_path)])

    cli.main()

    assert called["ran"] is True
    assert _RecordingPipeline.instances == []  # purge happened, no backfill
