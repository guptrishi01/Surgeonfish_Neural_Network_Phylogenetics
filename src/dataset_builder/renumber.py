"""One-time cleanup for the duplicate-index filenames an earlier bug left behind.

Before the fix in ``pipeline._next_available_index``, a backfill following a
mid-sequence rejection reused an index still in use by a surviving file
(e.g. both "024_gbif_A.jpg" and "024_gbif_B.jpg" could exist at once). No
data was lost - every file kept its own unique ``occurrence_key`` suffix -
but the numbering is confusing and unsuitable for a clean final dataset.
This renumbers each species' non-reference files to a gapless 001..N
sequence and updates the sourcing-log CSV to match.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REFERENCE_PREFIX = "000_reference"


def renumber_species_dir(species_dir: Path) -> dict[str, str]:
    """Renumbers non-reference files in one species folder to 001..N.

    Uses a two-phase rename (everything to a temporary name first) so that
    intermediate renames never collide with a file that hasn't been moved
    yet.

    Args:
        species_dir: A single species' image folder.

    Returns:
        Mapping of old filename to new filename, for every file actually
        renamed (files already correctly numbered are omitted).
    """
    files = sorted(
        p for p in species_dir.glob("*") if not p.stem.startswith(_REFERENCE_PREFIX)
    )
    renames: dict[str, str] = {}
    temp_paths: list[tuple[Path, str]] = []

    for i, path in enumerate(files, start=1):
        suffix = path.name.split("_gbif_", 1)[1] if "_gbif_" in path.name else path.name
        new_name = f"{i:03d}_gbif_{suffix}"
        if new_name == path.name:
            continue
        temp_path = species_dir / f".tmp_renumber_{i}{path.suffix}"
        path.rename(temp_path)
        temp_paths.append((temp_path, new_name))
        renames[path.name] = new_name

    for temp_path, new_name in temp_paths:
        temp_path.rename(species_dir / new_name)

    return renames


def update_metadata_csv(csv_path: Path, species_name: str, renames: dict[str, str]) -> None:
    """Rewrites filename references in the sourcing log to match a renumber.

    Args:
        csv_path: Path to reports/image_sourcing_log.csv.
        species_name: "Genus species" binomial the renames apply to.
        renames: Old filename -> new filename, as returned by
            renumber_species_dir.
    """
    if not csv_path.exists() or not renames:
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []

    changed = False
    for row in rows:
        if row.get("species") == species_name and row.get("filename") in renames:
            row["filename"] = renames[row["filename"]]
            changed = True

    if changed:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def renumber_all(raw_images_root: Path, metadata_csv_path: Path) -> dict[str, dict[str, str]]:
    """Renumbers every species folder under raw_images_root.

    Args:
        raw_images_root: Directory containing one subfolder per genus.
        metadata_csv_path: Sourcing log to keep in sync with the renames.

    Returns:
        Mapping of "Genus species" to its old->new filename renames, for
        every species that had at least one file renamed.
    """
    results = {}
    for genus_dir in sorted(p for p in raw_images_root.iterdir() if p.is_dir()):
        for species_dir in sorted(p for p in genus_dir.iterdir() if p.is_dir()):
            species_name = species_dir.name.replace("_", " ")
            renames = renumber_species_dir(species_dir)
            if renames:
                update_metadata_csv(metadata_csv_path, species_name, renames)
                results[species_name] = renames
                logger.info("%s: renumbered %d file(s)", species_name, len(renames))
    return results
