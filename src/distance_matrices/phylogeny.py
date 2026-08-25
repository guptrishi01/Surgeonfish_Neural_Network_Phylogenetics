"""Loads, prunes, and computes patristic distances from the reference phylogeny.

Built as the shared tree-loading utility both this phase's patristic
distance matrix and Phase 4's Kmult test need (Phase 4 hands the same
pruned tree object to R rather than reducing it to distances) - see the
package docstring.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from Bio import Phylo

from distance_matrices.species_coverage import load_matched_species

logger = logging.getLogger(__name__)


def resolve_tip_names(
    species_coverage_csv_path: Path, species_keys: list[str]
) -> dict[str, str]:
    """Maps each requested species to its tree tip label.

    Args:
        species_coverage_csv_path: The 64-species coverage table (provides
            the ``match_method`` column needed to resolve synonym cases).
        species_keys: "Genus_species" species to resolve - typically an
            aggregation run's actual output species list, which can be a
            subset of the full 50 matched species (e.g. after the
            reference-image policy drops *Naso maculatus*).

    Returns:
        species_key -> tip_name.

    Raises:
        AssertionError: If any requested species isn't in the coverage
            table's matched (has_genetic_data == "yes") set - a species
            that made it into an aggregate but isn't phylogeny-matched
            would be a real bug in the caller, not something to silently
            skip here.
    """
    matched = load_matched_species(species_coverage_csv_path)
    tip_name_by_species = {m.species_key: m.tip_name for m in matched}
    missing = set(species_keys) - set(tip_name_by_species)
    assert not missing, (
        f"{len(missing)} species not in the phylogeny-matched set: {sorted(missing)} - "
        "these shouldn't have been aggregated in the first place."
    )
    return {key: tip_name_by_species[key] for key in species_keys}


def load_pruned_tree(
    tree_path: Path,
    tip_names: set[str],
    min_tip_count: int,
    max_tip_count: int,
):
    """Loads the reference tree, asserts its tip count, and prunes to the given tips.

    Args:
        tree_path: Newick tree file (``data/phylogeny/actinopt_12k_treePL.tre``).
        tip_names: Exact tip labels to keep (see ``resolve_tip_names()``).
        min_tip_count: The loaded tree's tip count must exceed this.
        max_tip_count: The loaded tree's tip count must be under this.

    Returns:
        The pruned ``Bio.Phylo`` tree object.

    Raises:
        AssertionError: If the loaded tree's tip count falls outside
            [min_tip_count, max_tip_count] (wrong tree file - discriminates
            the genetic-data-only tree from the much larger "complete"
            tree, a range rather than an exact count so it doesn't break
            on a release differing by one tip), or if the pruned tree
            doesn't end up with exactly `tip_names` (a resolution or
            duplicate-tip-name problem).
    """
    tree = Phylo.read(tree_path, "newick")
    n_tips_before = tree.count_terminals()
    assert min_tip_count < n_tips_before < max_tip_count, (
        f"Loaded tree has {n_tips_before} tips, expected between {min_tip_count} and "
        f"{max_tip_count} - wrong tree file? (the much larger 'complete' tree uses "
        "stochastic polytomy resolution and should never be used here, see "
        "data/phylogeny/README.md)"
    )

    terminals = tree.get_terminals()
    to_prune = [t for t in terminals if t.name not in tip_names]
    # prune() removes one tip (and its now-redundant parent node) at a time;
    # pruning from a list computed up front, not re-querying get_terminals()
    # per iteration, since the tree mutates as tips are removed.
    for terminal in to_prune:
        tree.prune(terminal)

    remaining_names = {t.name for t in tree.get_terminals()}
    assert remaining_names == tip_names, (
        f"Pruned tree has {len(remaining_names)} tip(s), expected {len(tip_names)} - "
        f"missing: {sorted(tip_names - remaining_names)}, "
        f"unexpected: {sorted(remaining_names - tip_names)}"
    )

    logger.info("Pruned tree from %d to %d tips", n_tips_before, len(remaining_names))
    return tree


def patristic_distance_matrix(
    tree, species_order: list[str], tip_name_by_species: dict[str, str]
) -> np.ndarray:
    """Computes the pairwise patristic (branch-length-sum) distance matrix.

    Args:
        tree: A tree already pruned to exactly the tips in
            `tip_name_by_species` (e.g. from ``load_pruned_tree()``).
        species_order: species_key values defining the output matrix's
            row/column order - the canonical species labeling used
            throughout this package, not the tree's own tip names (which
            differ for synonym-resolved species).
        tip_name_by_species: species_key -> tip_name (from
            ``resolve_tip_names()``), used to look up each species' tree
            distance.

    Returns:
        (n, n) symmetric distance matrix, zero diagonal, in
        `species_order`.
    """
    terminals_by_tip_name = {t.name: t for t in tree.get_terminals()}
    n = len(species_order)
    matrix = np.zeros((n, n))
    clades = [terminals_by_tip_name[tip_name_by_species[species]] for species in species_order]
    for i in range(n):
        for j in range(i + 1, n):
            d = tree.distance(clades[i], clades[j])
            matrix[i, j] = matrix[j, i] = d
    return matrix
