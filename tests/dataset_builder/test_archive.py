"""Unit tests for per-species zip/unzip archiving."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from dataset_builder.archive import (
    unzip_all,
    unzip_species_dir,
    zip_all,
    zip_species_dir,
)


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), (0, 0, 0)).save(path)


def test_zip_species_dir_creates_zip_with_all_files_and_removes_source(tmp_path: Path):
    species_dir = tmp_path / "Zebrasoma_flavescens"
    _make_image(species_dir / "000_reference.jpg")
    _make_image(species_dir / "001_gbif_100.jpg")

    zip_path = zip_species_dir(species_dir)

    assert zip_path == tmp_path / "Zebrasoma_flavescens.zip"
    assert zip_path.exists()
    assert not species_dir.exists()  # source removed by default


def test_zip_species_dir_can_keep_source(tmp_path: Path):
    species_dir = tmp_path / "Zebrasoma_flavescens"
    _make_image(species_dir / "000_reference.jpg")

    zip_species_dir(species_dir, remove_source=False)

    assert species_dir.exists()


def test_zip_species_dir_raises_on_empty_folder(tmp_path: Path):
    species_dir = tmp_path / "Zebrasoma_flavescens"
    species_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        zip_species_dir(species_dir)


def test_unzip_species_dir_round_trips_exact_contents(tmp_path: Path):
    species_dir = tmp_path / "Zebrasoma_flavescens"
    _make_image(species_dir / "000_reference.jpg")
    _make_image(species_dir / "001_gbif_100.jpg")
    original_files = sorted(p.name for p in species_dir.glob("*"))

    zip_path = zip_species_dir(species_dir)
    restored_dir = unzip_species_dir(zip_path)

    assert sorted(p.name for p in restored_dir.glob("*")) == original_files
    assert zip_path.exists()  # remove_zip defaults to False


def test_unzip_species_dir_can_remove_zip_after_extracting(tmp_path: Path):
    species_dir = tmp_path / "Zebrasoma_flavescens"
    _make_image(species_dir / "000_reference.jpg")
    zip_path = zip_species_dir(species_dir)

    unzip_species_dir(zip_path, remove_zip=True)

    assert not zip_path.exists()


def test_zip_all_and_unzip_all_round_trip_a_small_dataset(tmp_path: Path):
    root = tmp_path / "raw_images"
    _make_image(root / "Acanthurus" / "Acanthurus_guttatus" / "000_reference.jpg")
    _make_image(root / "Acanthurus" / "Acanthurus_guttatus" / "001_gbif_1.jpg")
    _make_image(root / "Zebrasoma" / "Zebrasoma_flavescens" / "000_reference.jpg")

    zips = zip_all(root)
    assert len(zips) == 2
    assert not (root / "Acanthurus" / "Acanthurus_guttatus").exists()

    extracted = unzip_all(root)
    assert len(extracted) == 2
    assert sorted(p.name for p in (root / "Acanthurus" / "Acanthurus_guttatus").glob("*")) == [
        "000_reference.jpg",
        "001_gbif_1.jpg",
    ]
