"""Unit tests for the duplicate-index-filename cleanup utility."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from dataset_builder.renumber import (
    renumber_all,
    renumber_species_dir,
    update_metadata_csv,
)


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), (0, 0, 0)).save(path)


def test_renumber_leaves_already_clean_sequence_untouched(tmp_path: Path):
    species_dir = tmp_path / "Zebrasoma_flavescens"
    _make_image(species_dir / "000_reference.jpg")
    _make_image(species_dir / "001_gbif_100.jpg")
    _make_image(species_dir / "002_gbif_200.jpg")

    renames = renumber_species_dir(species_dir)

    assert renames == {}
    assert (species_dir / "001_gbif_100.jpg").exists()
    assert (species_dir / "002_gbif_200.jpg").exists()


def test_renumber_resolves_duplicate_index_prefixes(tmp_path: Path):
    species_dir = tmp_path / "Zebrasoma_flavescens"
    _make_image(species_dir / "000_reference.jpg")
    _make_image(species_dir / "001_gbif_100.jpg")
    _make_image(species_dir / "001_gbif_101.jpg")  # duplicate index (the bug)
    _make_image(species_dir / "003_gbif_300.jpg")

    renames = renumber_species_dir(species_dir)

    remaining = sorted(p.name for p in species_dir.glob("*_gbif_*"))
    assert remaining == ["001_gbif_100.jpg", "002_gbif_101.jpg", "003_gbif_300.jpg"]
    # Reference untouched, no data lost - every occurrence_key still present.
    assert (species_dir / "000_reference.jpg").exists()
    assert renames  # at least one file was actually renamed


def test_renumber_never_collides_mid_rename(tmp_path: Path):
    """Two-phase rename must handle a case where the target name of one
    file is the current name of another (e.g. 001 and 002 swapping)."""
    species_dir = tmp_path / "Zebrasoma_flavescens"
    _make_image(species_dir / "000_reference.jpg")
    # Duplicated "001" pushes everything after it up by one, so the file
    # currently named "002_gbif_..." must become "003_gbif_...".
    _make_image(species_dir / "001_gbif_100.jpg")
    _make_image(species_dir / "001_gbif_101.jpg")
    _make_image(species_dir / "002_gbif_200.jpg")

    renumber_species_dir(species_dir)

    remaining = sorted(p.name for p in species_dir.glob("*_gbif_*"))
    assert remaining == ["001_gbif_100.jpg", "002_gbif_101.jpg", "003_gbif_200.jpg"]


def test_update_metadata_csv_only_touches_matching_species(tmp_path: Path):
    csv_path = tmp_path / "log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["species", "filename", "license_url"])
        writer.writeheader()
        writer.writerow(
            {"species": "Zebrasoma flavescens", "filename": "001_gbif_101.jpg", "license_url": "cc"}
        )
        writer.writerow(
            {"species": "Naso annulatus", "filename": "001_gbif_101.jpg", "license_url": "cc"}
        )

    update_metadata_csv(csv_path, "Zebrasoma flavescens", {"001_gbif_101.jpg": "002_gbif_101.jpg"})

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    zeb_row = next(r for r in rows if r["species"] == "Zebrasoma flavescens")
    naso_row = next(r for r in rows if r["species"] == "Naso annulatus")
    assert zeb_row["filename"] == "002_gbif_101.jpg"
    assert naso_row["filename"] == "001_gbif_101.jpg"  # untouched - different species


def test_renumber_all_processes_every_species_and_updates_csv(tmp_path: Path):
    root = tmp_path / "raw_images"
    _make_image(root / "Zebrasoma" / "Zebrasoma_flavescens" / "000_reference.jpg")
    _make_image(root / "Zebrasoma" / "Zebrasoma_flavescens" / "001_gbif_100.jpg")
    _make_image(root / "Zebrasoma" / "Zebrasoma_flavescens" / "001_gbif_101.jpg")
    _make_image(root / "Acanthurus" / "Acanthurus_guttatus" / "000_reference.jpg")
    _make_image(root / "Acanthurus" / "Acanthurus_guttatus" / "001_gbif_500.jpg")

    csv_path = tmp_path / "log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["species", "filename"])
        writer.writeheader()
        writer.writerow({"species": "Zebrasoma flavescens", "filename": "001_gbif_101.jpg"})

    results = renumber_all(root, csv_path)

    assert "Zebrasoma flavescens" in results
    assert "Acanthurus guttatus" not in results  # already clean, nothing to do

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    assert rows[0]["filename"] == "002_gbif_101.jpg"
