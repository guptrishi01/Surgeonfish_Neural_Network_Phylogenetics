"""Exports prepared Kmult feature matrices and the pruned tree for R to consume."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from Bio import Phylo

from distance_matrices.phylogeny import load_pruned_tree, resolve_tip_names
from phylo_comparison.config import ExportConfig
from phylo_comparison.feature_prep import PreparedMatrix

logger = logging.getLogger(__name__)


def write_prepared_matrix_csv(prepared: PreparedMatrix, output_dir: Path) -> Path:
    """Writes one dimension's prepared feature matrix as a labeled CSV.

    Args:
        prepared: Output of ``feature_prep.prepare_dimension_matrix()``.
        output_dir: Directory to write into.

    Returns:
        The path written to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prepared.dimension}_kmult_features.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["species", *prepared.feature_names])
        for species, row in zip(prepared.species, prepared.matrix):
            writer.writerow([species, *[f"{v:.6f}" for v in row]])
    logger.info(
        "Wrote %s Kmult feature matrix (%d species x %d features, PAC.no=%d, "
        "condition number=%.3e) to %s",
        prepared.dimension, len(prepared.species), len(prepared.feature_names),
        prepared.pac_no, prepared.condition_number, output_path,
    )
    return output_path


def export_pruned_tree(config: ExportConfig, species_order: list[str]) -> Path:
    """Prunes the reference tree to `species_order` and writes it as Newick for R.

    Tip labels are renamed from the tree's own names (which differ from
    the canonical species_key for synonym-resolved species, e.g.
    "Zebrasoma_velifer" for "Zebrasoma_veliferum") to species_key before
    writing - `geomorph` matches feature-matrix rows to tree tips by name,
    so the exported tree's labels must agree exactly with the "species"
    column in every exported feature matrix CSV, not just have the same
    length/order.

    Args:
        config: Export settings.
        species_order: species_key values to prune to - must match the
            species in every prepared feature matrix exactly, so R's
            name-based join has something correct to match against.

    Returns:
        The path written to.
    """
    tip_name_by_species = resolve_tip_names(config.species_coverage_csv_path, species_order)
    species_by_tip_name = {tip: species for species, tip in tip_name_by_species.items()}

    tree = load_pruned_tree(
        config.tree_path, set(tip_name_by_species.values()),
        config.min_tip_count, config.max_tip_count,
    )
    for terminal in tree.get_terminals():
        terminal.name = species_by_tip_name[terminal.name]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "pruned_tree.nwk"
    Phylo.write(tree, output_path, "newick")
    logger.info("Wrote pruned tree (%d tips) to %s", len(species_order), output_path)
    return output_path
