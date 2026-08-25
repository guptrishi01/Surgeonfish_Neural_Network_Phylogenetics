"""Configuration objects for per-species aggregation, distance matrices, and phylogeny loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AggregationConfig:
    """Settings for collapsing per-image features into per-species vectors.

    Attributes:
        pattern_features_csv_path: Phase 2's per-image output
            (``pattern_extractor.pipeline``'s ``PipelineConfig.
            output_csv_path``).
        species_coverage_csv_path: The 64-species genetic-data-coverage
            table (``data/phylogeny/species_coverage.csv``) - used to
            restrict aggregation to the 50 species with real phylogenetic
            placement.
        extracted_root: Root of ``fish_extractor``'s output (mask-cutout
            crops), used to re-derive each image's total masked-pixel
            count for normalizing ``mean_spot_area`` - see the package
            docstring for why raw pixel area isn't comparable across
            images of different native resolutions.
        output_csv_path: Where the per-species aggregate feature table is
            written.
        include_reference_images: Whether to include each species'
            curated ``is_reference`` seed photo in its aggregate. Default
            False - see the package docstring for the reasoning and the
            *Naso maculatus* consequence (excluding it drops that species
            entirely, since its only image is the reference photo).
        min_images_per_species: Species with fewer than this many
            qualifying images (after the reference-image policy above is
            applied) are dropped from the aggregate rather than averaged
            from too little data. Default 1 (drop only species left with
            zero qualifying images); raise this for the sparse-species
            sensitivity check the README's Planned Approach step 3 calls
            for.
    """

    pattern_features_csv_path: Path = field(
        default_factory=lambda: Path("reports/pattern_features.csv")
    )
    species_coverage_csv_path: Path = field(
        default_factory=lambda: Path("data/phylogeny/species_coverage.csv")
    )
    extracted_root: Path = field(default_factory=lambda: Path("data/extracted_fish"))
    output_csv_path: Path = field(
        default_factory=lambda: Path("reports/species_features.csv")
    )
    include_reference_images: bool = False
    min_images_per_species: int = 1


@dataclass
class DistanceMatrixConfig:
    """Settings for building the three pattern-dimension distance matrices.

    Attributes:
        output_dir: Directory the three species x species CSVs
            (``color_distance_matrix.csv``, ``stripe_distance_matrix.csv``,
            ``spot_distance_matrix.csv``) are written into.
    """

    output_dir: Path = field(default_factory=lambda: Path("outputs"))


@dataclass
class PhylogenyConfig:
    """Settings for loading, pruning, and computing patristic distances from the reference tree.

    Attributes:
        tree_path: The genetic-data-only tree (``data/phylogeny/
            actinopt_12k_treePL.tre``), Newick format.
        species_coverage_csv_path: Same 64-species coverage table
            AggregationConfig uses - provides the ``match_method`` column
            (``exact`` or ``synonym (<tree tip name>)``) needed to resolve
            each of the 50 study species to its actual tree tip label,
            since one (*Zebrasoma veliferum* -> tree tip
            ``Zebrasoma_velifer``) doesn't match by exact name.
        min_tip_count: The loaded tree's tip count must exceed this.
        max_tip_count: The loaded tree's tip count must be under this.
            Together with min_tip_count, discriminates the genetic-data-
            only tree (11,638 tips) from the much larger "complete" tree
            (31,526 tips, stochastic-polytomy-resolved, deliberately not
            used - see ``data/phylogeny/README.md``) - a range check
            rather than an exact ``== 11638`` match, since the latter
            would break on any release differing by even one tip.
        output_path: Where the pruned patristic-distance matrix (species
            x species) is written.
    """

    tree_path: Path = field(
        default_factory=lambda: Path("data/phylogeny/actinopt_12k_treePL.tre")
    )
    species_coverage_csv_path: Path = field(
        default_factory=lambda: Path("data/phylogeny/species_coverage.csv")
    )
    min_tip_count: int = 10_000
    max_tip_count: int = 15_000
    output_path: Path = field(
        default_factory=lambda: Path("outputs/patristic_distance_matrix.csv")
    )
