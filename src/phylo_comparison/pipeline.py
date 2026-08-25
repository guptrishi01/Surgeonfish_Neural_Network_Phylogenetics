"""Orchestrates Phase 4's Python-side preparation.

Builds and exports every Kmult-ready feature matrix plus the pruned tree,
for ``r/phase4_kmult.R`` to consume. No resumable state - like Phase 3,
this is fast, deterministic, local computation (a CSV and a tree file, no
network or GPU calls), so a re-run simply recomputes and overwrites.
"""

from __future__ import annotations

import logging

from phylo_comparison.config import ExportConfig, FeaturePrepConfig
from phylo_comparison.export import export_pruned_tree, write_prepared_matrix_csv
from phylo_comparison.feature_prep import (
    DIMENSIONS,
    load_species_aggregates,
    prepare_dimension_matrix,
)

logger = logging.getLogger(__name__)


def run(feature_prep_config: FeaturePrepConfig, export_config: ExportConfig) -> list[str]:
    """Prepares and exports every dimension's Kmult-ready feature matrix, plus the pruned tree.

    Args:
        feature_prep_config: Feature-transform settings.
        export_config: Export/tree settings.

    Returns:
        The canonical species order used across every exported file.
    """
    aggregates = load_species_aggregates(feature_prep_config.species_features_csv_path)
    species_order = [a.species for a in aggregates]
    logger.info("Phase 4 preparation: %d species", len(species_order))

    for dimension in DIMENSIONS:
        prepared = prepare_dimension_matrix(aggregates, dimension, feature_prep_config)
        write_prepared_matrix_csv(prepared, export_config.output_dir)

    export_pruned_tree(export_config, species_order)
    return species_order
