"""Unit tests for resumable per-image extraction state."""

from __future__ import annotations

from pathlib import Path

from fish_extractor.state import ExtractionState, ImageExtractionState


def test_get_creates_fresh_pending_state_for_unseen_image(tmp_path: Path):
    state = ExtractionState(tmp_path / "state.json")

    entry = state.get("Acanthurus/Acanthurus_guttatus/000_reference.jpg")

    assert entry.status == "pending"
    assert entry.is_terminal is False


def test_is_processed_false_for_pending_and_true_for_terminal_statuses(tmp_path: Path):
    state = ExtractionState(tmp_path / "state.json")
    key = "Zebrasoma/Zebrasoma_flavescens/001_gbif_1.jpg"

    assert state.is_processed(key) is False

    state.get(key).status = "accepted"

    assert state.is_processed(key) is True


def test_save_and_reload_round_trips_all_fields(tmp_path: Path):
    path = tmp_path / "state.json"
    state = ExtractionState(path)
    key = "Naso/Naso_annulatus/002_gbif_2.jpg"
    entry = state.get(key)
    entry.status = "flagged"
    entry.reason = "multiple_fish"
    entry.box = [10.0, 20.0, 300.0, 400.0, 0.8]
    entry.center_offset_fraction = 0.12
    state.save()

    reloaded = ExtractionState(path)
    reloaded_entry = reloaded.get(key)

    assert reloaded_entry.status == "flagged"
    assert reloaded_entry.reason == "multiple_fish"
    assert reloaded_entry.box == [10.0, 20.0, 300.0, 400.0, 0.8]
    assert reloaded_entry.center_offset_fraction == 0.12


def test_excluded_status_is_terminal():
    entry = ImageExtractionState(status="excluded", reason="human_excluded")

    assert entry.is_terminal is True
