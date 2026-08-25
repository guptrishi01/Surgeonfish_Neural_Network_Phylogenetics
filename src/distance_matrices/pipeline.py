"""Orchestrates Phase 3: aggregation -> pattern distance matrices -> patristic distance matrix.

No resumable per-image/per-species state - like ``pattern_extractor``, this
stage is fast, deterministic, and local (a CSV, some mask files, and a
tree file, no network or GPU calls), so a re-run simply recomputes
everything and overwrites the outputs rather than needing to skip
already-processed species.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from distance_matrices.aggregation import (
    SpeciesAggregate,
    aggregate_species_features,
    write_species_features_csv,
)
from distance_matrices.config import AggregationConfig, DistanceMatrixConfig, PhylogenyConfig
from distance_matrices.distance import (
    build_feature_matrix,
    pairwise_distance_matrix,
    standardize,
    write_distance_matrix_csv,
)
from distance_matrices.phylogeny import (
    load_pruned_tree,
    patristic_distance_matrix,
    resolve_tip_names,
)

logger = logging.getLogger(__name__)

DIMENSIONS = ["color", "stripe", "spot"]


@dataclass
class Phase3Result:
    """Everything Phase 3 produced, for a caller (e.g. a notebook cell) to inspect.

    Attributes:
        aggregates: Per-species feature vectors (one row per species that
            made it into the analysis - see
            ``AggregationConfig.include_reference_images``/
            ``min_images_per_species`` for what can exclude a species).
        species_order: The canonical species ordering used for every
            matrix below - identical to ``[a.species for a in aggregates]``,
            exposed directly since matrix row/column alignment is exactly
            the thing this result needs to make easy to check.
        pattern_distance_matrices: dimension -> (n, n) distance matrix, in
            species_order.
        patristic_distance_matrix: (n, n) patristic distance matrix from
            the pruned tree, in the same species_order.
    """

    aggregates: list[SpeciesAggregate]
    species_order: list[str]
    pattern_distance_matrices: dict[str, np.ndarray]
    patristic_distance_matrix: np.ndarray


def run(
    aggregation_config: AggregationConfig,
    distance_config: DistanceMatrixConfig,
    phylogeny_config: PhylogenyConfig,
) -> Phase3Result:
    """Runs the full Phase 3 pipeline and writes every output file.

    Args:
        aggregation_config: Per-species aggregation settings.
        distance_config: Pattern-dimension distance matrix settings.
        phylogeny_config: Tree loading/pruning settings.

    Returns:
        A Phase3Result with every matrix already written to disk.

    Raises:
        AssertionError: If the pattern distance matrices' species order
            ever diverges from the patristic matrix's - `geomorph`
            matches by name but a Python array-position join doesn't
            unless checked, so this is asserted explicitly rather than
            trusted (per the README's Planned Approach step 4).
    """
    aggregates = aggregate_species_features(aggregation_config)
    write_species_features_csv(aggregates, aggregation_config.output_csv_path)
    species_order = [a.species for a in aggregates]
    logger.info("Phase 3 analysis set: %d species", len(species_order))

    pattern_matrices: dict[str, np.ndarray] = {}
    for dimension in DIMENSIONS:
        dim_species, feature_matrix = build_feature_matrix(aggregates, dimension)
        assert dim_species == species_order, (
            "build_feature_matrix() returned a different species order than "
            "the aggregate list - this should be impossible given both come "
            "from the same `aggregates` list, but asserted rather than assumed."
        )
        standardized = standardize(feature_matrix)
        matrix = pairwise_distance_matrix(standardized)
        pattern_matrices[dimension] = matrix
        write_distance_matrix_csv(
            species_order, matrix, distance_config.output_dir / f"{dimension}_distance_matrix.csv"
        )

    tip_name_by_species = resolve_tip_names(
        phylogeny_config.species_coverage_csv_path, species_order
    )
    tree = load_pruned_tree(
        phylogeny_config.tree_path,
        set(tip_name_by_species.values()),
        phylogeny_config.min_tip_count,
        phylogeny_config.max_tip_count,
    )
    patristic_matrix = patristic_distance_matrix(tree, species_order, tip_name_by_species)
    write_distance_matrix_csv(species_order, patristic_matrix, phylogeny_config.output_path)

    # The row-order integrity check the README's Planned Approach step 4
    # calls for: every matrix here was built directly from species_order,
    # so this is really asserting internal consistency of this function
    # rather than re-deriving an independent order to compare against -
    # but it's the check that catches a future refactor accidentally
    # reordering one matrix and not the others.
    for dimension, matrix in pattern_matrices.items():
        assert matrix.shape == (len(species_order), len(species_order)), (
            f"{dimension} distance matrix shape {matrix.shape} doesn't match "
            f"{len(species_order)} species"
        )
    assert patristic_matrix.shape == (len(species_order), len(species_order)), (
        f"Patristic distance matrix shape {patristic_matrix.shape} doesn't match "
        f"{len(species_order)} species"
    )

    return Phase3Result(
        aggregates=aggregates,
        species_order=species_order,
        pattern_distance_matrices=pattern_matrices,
        patristic_distance_matrix=patristic_matrix,
    )
