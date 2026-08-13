"""Unit tests for the one-time raw_images layout migration and per-species
fault isolation in the collection loop."""

from __future__ import annotations

from pathlib import Path

import requests

from dataset_builder.config import PipelineConfig
from dataset_builder.pipeline import (
    DatasetBuilderPipeline,
    _next_available_index,
    _slugify_species,
    migrate_existing_layout,
)


def test_slugify_species_strips_trailing_space_and_collapses_whitespace():
    assert _slugify_species("Acanthurus albipectoralis ") == "Acanthurus_albipectoralis"


def test_slugify_species_leaves_clean_underscore_name_untouched():
    assert _slugify_species("Acanthurus_achilles") == "Acanthurus_achilles"


def _make_source_image(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), (0, 0, 0)).save(path)


def test_migrate_moves_flat_files_into_species_subfolders(tmp_path: Path):
    root = tmp_path / "raw_images"
    _make_source_image(root / "Zebrasoma" / "Zebrasoma flavescens .jpeg")
    _make_source_image(root / "Acanthurus" / "Acanthurus_achilles.jpg")

    species_dirs = migrate_existing_layout(root)

    assert set(species_dirs) == {"Zebrasoma flavescens", "Acanthurus achilles"}
    zeb_dir = species_dirs["Zebrasoma flavescens"]
    assert zeb_dir == root / "Zebrasoma" / "Zebrasoma_flavescens"
    assert (zeb_dir / "000_reference.jpeg").exists()
    # Original flat file must be gone, not copied.
    assert not (root / "Zebrasoma" / "Zebrasoma flavescens .jpeg").exists()


def test_migrate_is_idempotent(tmp_path: Path):
    root = tmp_path / "raw_images"
    _make_source_image(root / "Zebrasoma" / "Zebrasoma flavescens .jpeg")

    first_pass = migrate_existing_layout(root)
    second_pass = migrate_existing_layout(root)

    assert first_pass.keys() == second_pass.keys()
    zeb_dir = second_pass["Zebrasoma flavescens"]
    # Still exactly one reference file - no duplicate/overwrite churn.
    assert list(zeb_dir.glob("*")) == [zeb_dir / "000_reference.jpeg"]


def test_migrate_skips_non_image_files(tmp_path: Path):
    root = tmp_path / "raw_images"
    genus_dir = root / "Zebrasoma"
    genus_dir.mkdir(parents=True)
    (genus_dir / "notes.txt").write_text("not an image")

    species_dirs = migrate_existing_layout(root)

    assert species_dirs == {}
    assert (genus_dir / "notes.txt").exists()


def test_next_available_index_on_empty_species_dir(tmp_path: Path):
    species_dir = tmp_path / "Zebrasoma_flavescens"
    species_dir.mkdir()
    _make_source_image(species_dir / "000_reference.jpg")

    assert _next_available_index(species_dir) == 0


def test_next_available_index_after_middle_removal_does_not_collide(tmp_path: Path):
    """Regression test: a real run rejected + removed a mid-sequence file
    (e.g. 010 of 000-024), then backfilled a replacement. The old logic
    computed the next index from the remaining file *count* (24), which
    collided with the still-present 024 file, producing duplicate-prefixed
    filenames like both "024_gbif_A.jpg" and "024_gbif_B.jpg" in the same
    folder. The fix scans actual filenames for the true max index instead.
    """
    species_dir = tmp_path / "Zebrasoma_flavescens"
    species_dir.mkdir()
    _make_source_image(species_dir / "000_reference.jpg")
    # Files 001-024 minus 010 (simulating a rejected-and-removed mid-sequence file).
    for i in range(1, 25):
        if i == 10:
            continue
        _make_source_image(species_dir / f"{i:03d}_gbif_{1000 + i}.jpg")

    # 23 files present (24 minus the removed one), but the true next index
    # must come after the highest surviving index (024), not the count (23).
    assert len(list(species_dir.glob("*_gbif_*"))) == 23
    assert _next_available_index(species_dir) == 25


class _FaultySpeciesGBIF:
    """A GBIF client whose first species always fails with a network error,
    to test that one species' persistent failure doesn't abort the run."""

    def match_taxon_key(self, scientific_name: str) -> int | None:
        if scientific_name == "Acanthurus achilles":
            raise requests.ConnectionError("simulated persistent outage")
        return 1

    def iter_media(self, taxon_key: int, seen_occurrence_keys: set[int]):
        return iter(())  # no candidates - species finishes "exhausted" immediately


def test_run_continues_past_a_species_that_persistently_fails(tmp_path: Path):
    root = tmp_path / "raw_images"
    _make_source_image(root / "Acanthurus" / "Acanthurus achilles.jpg")
    _make_source_image(root / "Zebrasoma" / "Zebrasoma flavescens.jpg")

    config = PipelineConfig(
        raw_images_root=root,
        target_per_species=5,
        state_path=tmp_path / "state.json",
        metadata_csv_path=tmp_path / "sourcing_log.csv",
    )
    pipeline = DatasetBuilderPipeline(config)
    pipeline._gbif = _FaultySpeciesGBIF()

    results = pipeline.run()

    by_species = {r.species_name: r for r in results}
    assert len(results) == 2  # both species get a result, none crash the run
    assert not by_species["Acanthurus achilles"].met_target
    assert not by_species["Zebrasoma flavescens"].met_target  # exhausted at 1/5, no candidates
