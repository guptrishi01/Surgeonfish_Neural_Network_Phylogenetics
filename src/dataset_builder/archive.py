"""Per-species zip archiving for data/raw_images/.

Tracking ~1,400 individual image files in git is unnecessary churn once a
species' collection is stable - one zip per species collapses that to one
tracked file per species while keeping the same per-species processing
unit every other pipeline stage already uses. Phase 1 (standardization)
unzips on demand rather than assuming loose files are present on disk.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def zip_species_dir(species_dir: Path, *, remove_source: bool = True) -> Path:
    """Zips one species' image folder and (by default) removes the loose files.

    Args:
        species_dir: A single species' image folder (e.g.
            data/raw_images/Acanthurus/Acanthurus_guttatus/).
        remove_source: If True, deletes the loose files/folder after the
            zip is written and verified.

    Returns:
        Path to the created zip file, named <species_dir>.zip alongside
        the (now-removed, if remove_source) source folder.

    Raises:
        FileNotFoundError: If species_dir doesn't exist or is empty.
    """
    files = sorted(p for p in species_dir.glob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files to zip in {species_dir}")

    zip_path = species_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.name)

    with zipfile.ZipFile(zip_path) as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            zip_path.unlink()
            raise OSError(f"Corrupt entry in {zip_path}: {bad_file}")
        if sorted(zf.namelist()) != sorted(p.name for p in files):
            zip_path.unlink()
            raise OSError(f"Zip contents don't match source files for {species_dir}")

    if remove_source:
        shutil.rmtree(species_dir)

    logger.info("Zipped %d file(s) from %s -> %s", len(files), species_dir, zip_path)
    return zip_path


def unzip_species_dir(zip_path: Path, *, remove_zip: bool = False) -> Path:
    """Extracts a species zip back into a loose-files folder.

    Args:
        zip_path: Path to a per-species zip (e.g.
            data/raw_images/Acanthurus/Acanthurus_guttatus.zip).
        remove_zip: If True, deletes the zip after successful extraction.

    Returns:
        Path to the extracted species folder (same stem as the zip).
    """
    species_dir = zip_path.with_suffix("")
    species_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(species_dir)

    if remove_zip:
        zip_path.unlink()

    logger.info("Unzipped %s -> %s", zip_path, species_dir)
    return species_dir


def zip_all(raw_images_root: Path) -> list[Path]:
    """Zips every species folder under raw_images_root.

    Args:
        raw_images_root: Directory containing one subfolder per genus.

    Returns:
        Paths to every zip file created.
    """
    created = []
    for genus_dir in sorted(p for p in raw_images_root.iterdir() if p.is_dir()):
        for species_dir in sorted(p for p in genus_dir.iterdir() if p.is_dir()):
            created.append(zip_species_dir(species_dir))
    return created


def unzip_all(raw_images_root: Path) -> list[Path]:
    """Unzips every species zip under raw_images_root.

    Args:
        raw_images_root: Directory containing one subfolder per genus.

    Returns:
        Paths to every species folder extracted.
    """
    extracted = []
    for genus_dir in sorted(p for p in raw_images_root.iterdir() if p.is_dir()):
        for zip_path in sorted(genus_dir.glob("*.zip")):
            extracted.append(unzip_species_dir(zip_path))
    return extracted
