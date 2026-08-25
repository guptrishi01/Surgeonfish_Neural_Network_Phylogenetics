"""Unit tests for the shared species-coverage loader."""

from __future__ import annotations

import csv
from pathlib import Path

from distance_matrices.species_coverage import load_matched_species

_FIELDNAMES = ["species", "genus", "has_genetic_data", "match_method"]


def _write_coverage_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_only_matched_species_are_returned(tmp_path: Path):
    path = tmp_path / "species_coverage.csv"
    _write_coverage_csv(path, [
        {"species": "Acanthurus achilles", "genus": "Acanthurus",
         "has_genetic_data": "yes", "match_method": "exact"},
        {"species": "Acanthurus albimento", "genus": "Acanthurus",
         "has_genetic_data": "no", "match_method": ""},
    ])

    matched = load_matched_species(path)

    assert [m.species_key for m in matched] == ["Acanthurus_achilles"]


def test_exact_match_uses_species_key_as_tip_name(tmp_path: Path):
    path = tmp_path / "species_coverage.csv"
    _write_coverage_csv(path, [
        {"species": "Acanthurus achilles", "genus": "Acanthurus",
         "has_genetic_data": "yes", "match_method": "exact"},
    ])

    matched = load_matched_species(path)

    assert matched[0].species_key == "Acanthurus_achilles"
    assert matched[0].tip_name == "Acanthurus_achilles"


def test_synonym_match_resolves_to_the_recorded_tip_name(tmp_path: Path):
    path = tmp_path / "species_coverage.csv"
    _write_coverage_csv(path, [
        {"species": "Zebrasoma veliferum", "genus": "Zebrasoma", "has_genetic_data": "yes",
         "match_method": "synonym (Zebrasoma_velifer)"},
    ])

    matched = load_matched_species(path)

    assert matched[0].species_key == "Zebrasoma_veliferum"
    assert matched[0].tip_name == "Zebrasoma_velifer"


def test_results_sorted_by_species_key(tmp_path: Path):
    path = tmp_path / "species_coverage.csv"
    _write_coverage_csv(path, [
        {"species": "Zebrasoma scopas", "genus": "Zebrasoma",
         "has_genetic_data": "yes", "match_method": "exact"},
        {"species": "Acanthurus achilles", "genus": "Acanthurus",
         "has_genetic_data": "yes", "match_method": "exact"},
    ])

    matched = load_matched_species(path)

    assert [m.species_key for m in matched] == ["Acanthurus_achilles", "Zebrasoma_scopas"]
