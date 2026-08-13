"""Resumable per-species progress tracking for the dataset-building pipeline.

Re-running the pipeline must not re-download or re-evaluate a candidate
that was already accepted or rejected in a prior run. All of that history
lives in one JSON file, loaded at startup and saved after each species
finishes its pass.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SpeciesState:
    """Progress for a single species.

    Attributes:
        taxon_key: Cached GBIF usageKey, so re-runs skip the name-match call.
        accepted_count: How many images currently sit in this species'
            folder (seed reference image included).
        seen_occurrence_keys: GBIF occurrence keys already evaluated
            (accepted or rejected), never re-fetched.
        accepted_hashes: Average hashes of accepted images, for cross-run
            near-duplicate detection.
        exhausted: True once GBIF has no more candidates to offer, or the
            per-species candidate cap was hit, short of target.
    """

    taxon_key: int | None = None
    accepted_count: int = 0
    seen_occurrence_keys: set[int] = field(default_factory=set)
    accepted_hashes: list[int] = field(default_factory=list)
    exhausted: bool = False


class PipelineState:
    """In-memory + on-disk store of every species' SpeciesState."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._species: dict[str, SpeciesState] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        for name, entry in raw.items():
            self._species[name] = SpeciesState(
                taxon_key=entry.get("taxon_key"),
                accepted_count=entry.get("accepted_count", 0),
                seen_occurrence_keys=set(entry.get("seen_occurrence_keys", [])),
                accepted_hashes=entry.get("accepted_hashes", []),
                exhausted=entry.get("exhausted", False),
            )
        logger.info("Loaded pipeline state for %d species from %s", len(self._species), self._path)

    def save(self) -> None:
        """Writes the full state to disk, overwriting any prior file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            name: {
                "taxon_key": s.taxon_key,
                "accepted_count": s.accepted_count,
                "seen_occurrence_keys": sorted(s.seen_occurrence_keys),
                "accepted_hashes": s.accepted_hashes,
                "exhausted": s.exhausted,
            }
            for name, s in self._species.items()
        }
        self._path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    def get(self, species_name: str) -> SpeciesState:
        """Returns the state for a species, creating a fresh one if absent."""
        if species_name not in self._species:
            self._species[species_name] = SpeciesState()
        return self._species[species_name]
