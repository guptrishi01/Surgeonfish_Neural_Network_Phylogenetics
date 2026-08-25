"""Shared loader for the 64-species genetic-data-coverage table.

Used by both ``aggregation.py`` (to restrict per-species feature
aggregation to species with real phylogenetic placement) and
``phylogeny.py`` (to resolve each of those species to its actual tree tip
label, since one - *Zebrasoma veliferum* - only matches the tree via a
recorded synonym, not its own name).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MatchedSpecies:
    """One species with real genetic-data phylogenetic placement.

    Attributes:
        species_key: "Genus_species" (underscored), matching
            `pattern_extractor`'s image_key species-folder naming
            convention - the canonical species label used throughout
            this package's output files.
        tip_name: The exact tip label this species resolves to in the
            reference tree - equal to species_key for an exact match, or
            the tree's own tip label for a resolved synonym (e.g.
            "Zebrasoma_velifer" for "Zebrasoma_veliferum").
    """

    species_key: str
    tip_name: str


def load_matched_species(species_coverage_csv_path: Path) -> list[MatchedSpecies]:
    """Loads the subset of the 64-species coverage table with real genetic-data placement.

    Args:
        species_coverage_csv_path: Path to
            ``data/phylogeny/species_coverage.csv`` (columns: species,
            genus, has_genetic_data, match_method).

    Returns:
        One MatchedSpecies per row where ``has_genetic_data == "yes"``,
        sorted by species_key for a deterministic, reproducible ordering
        used as the canonical species order across this package's outputs.
    """
    with open(species_coverage_csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    matched = []
    for row in rows:
        if row["has_genetic_data"] != "yes":
            continue
        species_key = row["species"].replace(" ", "_")
        match_method = row["match_method"]
        if match_method == "exact":
            tip_name = species_key
        else:
            # e.g. "synonym (Zebrasoma_velifer)"
            tip_name = match_method.split("(", 1)[1].rstrip(")")
        matched.append(MatchedSpecies(species_key=species_key, tip_name=tip_name))

    return sorted(matched, key=lambda m: m.species_key)
