"""Unit tests for resumable per-species pipeline state."""

from __future__ import annotations

from pathlib import Path

from dataset_builder.state import PipelineState


def test_get_creates_fresh_state_for_unknown_species(tmp_path: Path):
    state = PipelineState(tmp_path / "state.json")
    species_state = state.get("Zebrasoma flavescens")
    assert species_state.accepted_count == 0
    assert species_state.taxon_key is None
    assert species_state.seen_occurrence_keys == set()
    assert not species_state.exhausted


def test_get_returns_same_object_across_calls(tmp_path: Path):
    state = PipelineState(tmp_path / "state.json")
    state.get("Zebrasoma flavescens").accepted_count = 5
    assert state.get("Zebrasoma flavescens").accepted_count == 5


def test_save_and_reload_round_trips_all_fields(tmp_path: Path):
    path = tmp_path / "state.json"
    state = PipelineState(path)
    species_state = state.get("Zebrasoma flavescens")
    species_state.taxon_key = 2379869
    species_state.accepted_count = 3
    species_state.seen_occurrence_keys = {111, 222}
    species_state.accepted_hashes = [123456, 654321]
    species_state.exhausted = True
    state.save()

    reloaded = PipelineState(path)
    reloaded_state = reloaded.get("Zebrasoma flavescens")
    assert reloaded_state.taxon_key == 2379869
    assert reloaded_state.accepted_count == 3
    assert reloaded_state.seen_occurrence_keys == {111, 222}
    assert reloaded_state.accepted_hashes == [123456, 654321]
    assert reloaded_state.exhausted is True


def test_reload_of_missing_file_starts_empty(tmp_path: Path):
    state = PipelineState(tmp_path / "does_not_exist.json")
    assert state.get("Naso annulatus").accepted_count == 0
