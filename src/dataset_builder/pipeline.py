"""Orchestrates balanced, resumable image collection for every species.

One-time layout migration (flat "Genus species.ext" files -> one subfolder
per species, each holding its original reference photo plus newly-sourced
ones) is handled here too, since it only needs to run once before the
collection loop can operate per-species.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

from dataset_builder.config import PipelineConfig
from dataset_builder.gbif_client import GBIFClient
from dataset_builder.quality_filter import average_hash, evaluate
from dataset_builder.state import PipelineState

logger = logging.getLogger(__name__)

_KNOWN_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_REFERENCE_PREFIX = "000_reference"


@dataclass
class SpeciesResult:
    """Outcome of running collection for one species.

    Attributes:
        species_name: "Genus species" binomial.
        accepted_count: Final image count in the species' folder.
        target: The balanced-count goal.
        met_target: Whether accepted_count reached target.
    """

    species_name: str
    accepted_count: int
    target: int
    met_target: bool


def _slugify_species(raw_name: str) -> str:
    """Normalizes a filename stem to a clean "Genus_species" folder slug."""
    collapsed = re.sub(r"[\s_]+", "_", raw_name.strip())
    return collapsed.strip("_")


def migrate_existing_layout(raw_images_root: Path) -> dict[str, Path]:
    """Moves each genus' flat reference images into per-species subfolders.

    Idempotent: a species whose subfolder already exists is left untouched.
    The original image file itself is not renamed or altered - only moved -
    so any prior filename inconsistencies are preserved as-is, deliberately
    out of scope here.

    Args:
        raw_images_root: Directory containing one subfolder per genus.

    Returns:
        Mapping of "Genus species" binomial to its species subfolder path,
        for every genus/species found under raw_images_root.
    """
    species_dirs: dict[str, Path] = {}
    for genus_dir in sorted(p for p in raw_images_root.iterdir() if p.is_dir()):
        for entry in sorted(genus_dir.iterdir()):
            if entry.is_dir():
                species_name = entry.name.replace("_", " ")
                species_dirs[species_name] = entry
                continue
            if entry.suffix.lower() not in _KNOWN_IMAGE_SUFFIXES:
                continue
            slug = _slugify_species(entry.stem)
            species_dir = genus_dir / slug
            species_dir.mkdir(exist_ok=True)
            destination = species_dir / f"{_REFERENCE_PREFIX}{entry.suffix.lower()}"
            if not destination.exists():
                entry.rename(destination)
                logger.info("Migrated %s -> %s", entry, destination)
            species_dirs[slug.replace("_", " ")] = species_dir
    return species_dirs


def _guess_extension(image_url: str) -> str:
    suffix = Path(urlparse(image_url).path).suffix.lower()
    return suffix if suffix in _KNOWN_IMAGE_SUFFIXES else ".jpg"


_INDEX_PREFIX = re.compile(r"^(\d+)_gbif_")


def _next_available_index(species_dir: Path) -> int:
    """Finds the next unused numeric filename prefix in a species folder.

    Using ``state.accepted_count`` for this (as an earlier version did)
    breaks as soon as a file is removed from the middle of the sequence
    (e.g. a rejected image) rather than the end: the count drops but
    existing files keep their original index, so the next backfilled file
    collides with an index already in use. Scanning actual filenames for
    the true maximum avoids that regardless of what's been removed.
    """
    max_index = -1
    for path in species_dir.glob("*_gbif_*"):
        match = _INDEX_PREFIX.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


class DatasetBuilderPipeline:
    """Collects a balanced, licensed image set per species via GBIF."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._gbif = GBIFClient(config.gbif)
        self._state = PipelineState(config.state_path)

    def _seed_existing_images(self, species_name: str, species_dir: Path) -> None:
        """Registers images already on disk (the reference photo, or images
        from a prior run) against this species' accepted count/hashes."""
        state = self._state.get(species_name)
        if state.accepted_count > 0:
            return  # already seeded in a prior run
        existing = sorted(species_dir.glob("*"))
        for path in existing:
            try:
                image = Image.open(path)
                image.load()
            except Exception:  # noqa: BLE001
                continue
            state.accepted_hashes.append(average_hash(image))
        state.accepted_count = len(existing)

    def _write_metadata_row(self, species_name: str, filename: str, *, source_url: str,
                             license_url: str, rights_holder: str, occurrence_key: int,
                             width: int, height: int, border_edge_density: float) -> None:
        csv_path = self._config.metadata_csv_path
        is_new = not csv_path.exists()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow([
                    "species", "filename", "source_url", "license_url",
                    "rights_holder", "occurrence_key", "width", "height",
                    "border_edge_density",
                ])
            writer.writerow([
                species_name, filename, source_url, license_url,
                rights_holder, occurrence_key, width, height,
                round(border_edge_density, 4),
            ])

    def _collect_for_species(self, species_name: str, species_dir: Path) -> SpeciesResult:
        target = self._config.target_per_species
        state = self._state.get(species_name)
        self._seed_existing_images(species_name, species_dir)

        if state.accepted_count >= target:
            logger.info("%s: already at target (%d/%d)", species_name, state.accepted_count, target)
            return SpeciesResult(species_name, state.accepted_count, target, True)

        if state.exhausted:
            logger.warning(
                "%s: marked exhausted in a prior run, %d/%d - skipping",
                species_name, state.accepted_count, target,
            )
            return SpeciesResult(species_name, state.accepted_count, target, False)

        if state.taxon_key is None:
            state.taxon_key = self._gbif.match_taxon_key(species_name)
            if state.taxon_key is None:
                state.exhausted = True
                self._state.save()
                return SpeciesResult(species_name, state.accepted_count, target, False)

        candidates_evaluated = 0
        next_index = _next_available_index(species_dir)
        for media in self._gbif.iter_media(state.taxon_key, state.seen_occurrence_keys):
            if state.accepted_count >= target:
                break
            candidates_evaluated += 1
            if candidates_evaluated > self._config.max_candidates_per_species:
                logger.warning(
                    "%s: hit candidate cap (%d) before reaching target, %d/%d",
                    species_name, self._config.max_candidates_per_species,
                    state.accepted_count, target,
                )
                break

            state.seen_occurrence_keys.add(media.occurrence_key)
            image_bytes = self._gbif.download(
                media.image_url, self._config.quality.max_file_size_bytes
            )
            if image_bytes is None:
                continue

            result = evaluate(image_bytes, state.accepted_hashes, self._config.quality)
            if not result.accepted:
                logger.debug("%s: rejected (%s) %s", species_name, result.reason, media.image_url)
                continue

            extension = _guess_extension(media.image_url)
            filename = f"{next_index:03d}_gbif_{media.occurrence_key}{extension}"
            (species_dir / filename).write_bytes(image_bytes)
            self._write_metadata_row(
                species_name, filename,
                source_url=media.image_url, license_url=media.license_url,
                rights_holder=media.rights_holder, occurrence_key=media.occurrence_key,
                width=result.width, height=result.height,
                border_edge_density=result.border_edge_density,
            )
            state.accepted_hashes.append(result.average_hash)
            state.accepted_count += 1
            next_index += 1
            logger.info(
                "%s: accepted %d/%d (%s)", species_name, state.accepted_count, target, filename
            )
        else:
            # Generator exhausted without hitting target or the candidate cap.
            if state.accepted_count < target:
                state.exhausted = True

        self._state.save()
        met = state.accepted_count >= target
        if not met:
            logger.warning("%s: finished short, %d/%d", species_name, state.accepted_count, target)
        return SpeciesResult(species_name, state.accepted_count, target, met)

    def run(self, species_filter: set[str] | None = None) -> list[SpeciesResult]:
        """Runs collection for every species (or a filtered subset).

        Args:
            species_filter: If given, only these "Genus species" binomials
                are processed - useful for a small pilot run.

        Returns:
            One SpeciesResult per species processed.
        """
        species_dirs = migrate_existing_layout(self._config.raw_images_root)
        results = []
        for species_name in sorted(species_dirs):
            if species_filter is not None and species_name not in species_filter:
                continue
            try:
                results.append(
                    self._collect_for_species(species_name, species_dirs[species_name])
                )
            except requests.RequestException as exc:
                # Retries are already exhausted by this point (GBIFClient
                # retries transient failures itself) - a persistent network
                # or GBIF-outage problem on one species shouldn't forfeit
                # progress already made on the other ~60. Whatever state
                # was saved before this failure (via _state.save() calls
                # inside _collect_for_species) is kept; re-running the CLI
                # will resume this species from there.
                logger.error("%s: request failed, skipping for this run: %s", species_name, exc)
                state = self._state.get(species_name)
                target = self._config.target_per_species
                results.append(SpeciesResult(species_name, state.accepted_count, target, False))
        return results
