"""Configuration objects for Kmult feature-matrix preparation and export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeaturePrepConfig:
    """Settings for preparing Phase 3's per-species features for Kmult.

    Attributes:
        species_features_csv_path: Phase 3's per-species aggregate table
            (``distance_matrices.config.AggregationConfig.output_csv_path``).
        condition_number_threshold: A prepared feature matrix with a
            condition number at or above this is treated as ill-
            conditioned and rejected (``ValueError``) rather than handed
            to R to produce a silently-unstable result. 1e10 is a common
            rule-of-thumb cutoff (double-precision floating point has
            about 15-16 significant decimal digits; a condition number
            near 1e15-1e16 would leave effectively no precision at all),
            chosen loosely and not yet validated against this specific
            feature set - flagged as a reasoned interim value, same
            caveat as `pattern_extractor`'s calibrated thresholds carried
            before real-data validation.
    """

    species_features_csv_path: Path = field(
        default_factory=lambda: Path("reports/species_features.csv")
    )
    condition_number_threshold: float = 1e10


@dataclass
class ExportConfig:
    """Settings for exporting prepared feature matrices and the pruned tree for R.

    Attributes:
        output_dir: Directory the per-dimension prepared feature matrix
            CSVs and the pruned tree Newick file are written into.
        tree_path: The genetic-data-only reference tree (same file
            ``distance_matrices.config.PhylogenyConfig.tree_path`` points
            to).
        species_coverage_csv_path: The 64-species coverage table (same
            file used throughout Phase 3).
        min_tip_count: Passed through to ``distance_matrices.phylogeny``'s
            tip-count assertion.
        max_tip_count: Passed through to ``distance_matrices.phylogeny``'s
            tip-count assertion.
    """

    output_dir: Path = field(default_factory=lambda: Path("outputs/phase4"))
    tree_path: Path = field(
        default_factory=lambda: Path("data/phylogeny/actinopt_12k_treePL.tre")
    )
    species_coverage_csv_path: Path = field(
        default_factory=lambda: Path("data/phylogeny/species_coverage.csv")
    )
    min_tip_count: int = 10_000
    max_tip_count: int = 15_000
