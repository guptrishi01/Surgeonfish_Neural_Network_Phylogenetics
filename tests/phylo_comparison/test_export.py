"""Unit tests for exporting prepared feature matrices and the pruned tree."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from Bio import Phylo

from phylo_comparison.config import ExportConfig
from phylo_comparison.export import export_pruned_tree, write_prepared_matrix_csv
from phylo_comparison.feature_prep import PreparedMatrix


def _write_coverage_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["species", "genus", "has_genetic_data", "match_method"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_write_prepared_matrix_csv_round_trips(tmp_path: Path):
    prepared = PreparedMatrix(
        dimension="color",
        species=["A", "B"],
        feature_names=["f1", "f2"],
        matrix=np.array([[0.1, 0.2], [0.3, 0.4]]),
        pac_no=2,
        condition_number=1.5,
    )

    path = write_prepared_matrix_csv(prepared, tmp_path)

    assert path.name == "color_kmult_features.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["species", "f1", "f2"]
    assert rows[1][0] == "A"
    assert float(rows[1][1]) == 0.1


def test_export_pruned_tree_renames_tips_to_species_key(tmp_path: Path):
    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text("(Acanthurus_achilles:1,Zebrasoma_velifer:1);", encoding="utf-8")

    coverage_path = tmp_path / "species_coverage.csv"
    _write_coverage_csv(coverage_path, [
        {"species": "Acanthurus achilles", "genus": "Acanthurus",
         "has_genetic_data": "yes", "match_method": "exact"},
        {"species": "Zebrasoma veliferum", "genus": "Zebrasoma",
         "has_genetic_data": "yes", "match_method": "synonym (Zebrasoma_velifer)"},
    ])

    config = ExportConfig(
        output_dir=tmp_path / "outputs",
        tree_path=tree_path,
        species_coverage_csv_path=coverage_path,
        min_tip_count=1,
        max_tip_count=10,
    )

    output_path = export_pruned_tree(config, ["Acanthurus_achilles", "Zebrasoma_veliferum"])

    written_tree = Phylo.read(output_path, "newick")
    tip_names = {t.name for t in written_tree.get_terminals()}
    # The tree's own tip was "Zebrasoma_velifer" - the exported file must
    # use the species_key "Zebrasoma_veliferum" instead, matching the
    # feature matrix CSVs' "species" column exactly.
    assert tip_names == {"Acanthurus_achilles", "Zebrasoma_veliferum"}
