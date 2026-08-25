"""Unit tests for phylogeny loading, pruning, and patristic distance computation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from distance_matrices.phylogeny import (
    load_pruned_tree,
    patristic_distance_matrix,
    resolve_tip_names,
)


def _write_tree(path: Path, newick: str) -> None:
    path.write_text(newick, encoding="utf-8")


def test_load_pruned_tree_rejects_tip_count_outside_range(tmp_path: Path):
    tree_path = tmp_path / "tree.nwk"
    _write_tree(tree_path, "(A:1,B:1,C:1);")

    with pytest.raises(AssertionError):
        load_pruned_tree(tree_path, {"A", "B"}, min_tip_count=10, max_tip_count=100)


def test_load_pruned_tree_keeps_only_requested_tips(tmp_path: Path):
    tree_path = tmp_path / "tree.nwk"
    _write_tree(tree_path, "((A:1,B:1):1,(C:1,D:1):1);")

    tree = load_pruned_tree(tree_path, {"A", "C"}, min_tip_count=2, max_tip_count=10)

    assert {t.name for t in tree.get_terminals()} == {"A", "C"}


def test_load_pruned_tree_raises_if_a_requested_tip_is_missing(tmp_path: Path):
    tree_path = tmp_path / "tree.nwk"
    _write_tree(tree_path, "(A:1,B:1);")

    with pytest.raises(AssertionError):
        load_pruned_tree(tree_path, {"A", "NotInTree"}, min_tip_count=1, max_tip_count=10)


def test_patristic_distance_matrix_matches_known_branch_lengths(tmp_path: Path):
    # A---1---root---1---B: patristic distance A-B should be 1+1=2
    tree_path = tmp_path / "tree.nwk"
    _write_tree(tree_path, "(A:1,B:1);")
    tree = load_pruned_tree(tree_path, {"A", "B"}, min_tip_count=1, max_tip_count=10)

    matrix = patristic_distance_matrix(tree, ["A", "B"], {"A": "A", "B": "B"})

    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == pytest.approx(2.0)
    assert matrix[0, 0] == 0.0


def test_patristic_distance_matrix_uses_species_order_for_row_labeling(tmp_path: Path):
    tree_path = tmp_path / "tree.nwk"
    _write_tree(tree_path, "(A:1,B:2);")
    tree = load_pruned_tree(tree_path, {"A", "B"}, min_tip_count=1, max_tip_count=10)

    # species_order reversed relative to the tree's own tip order
    matrix = patristic_distance_matrix(tree, ["B", "A"], {"A": "A", "B": "B"})

    assert matrix[0, 1] == pytest.approx(3.0)  # still the A-B distance, symmetric either way


def test_resolve_tip_names_maps_species_to_tree_tip_labels(tmp_path: Path):
    coverage_path = tmp_path / "species_coverage.csv"
    with open(coverage_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["species", "genus", "has_genetic_data", "match_method"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "species": "Zebrasoma veliferum", "genus": "Zebrasoma",
            "has_genetic_data": "yes", "match_method": "synonym (Zebrasoma_velifer)",
        })

    result = resolve_tip_names(coverage_path, ["Zebrasoma_veliferum"])

    assert result == {"Zebrasoma_veliferum": "Zebrasoma_velifer"}


def test_resolve_tip_names_raises_for_an_unmatched_species(tmp_path: Path):
    coverage_path = tmp_path / "species_coverage.csv"
    with open(coverage_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["species", "genus", "has_genetic_data", "match_method"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "species": "Acanthurus achilles", "genus": "Acanthurus",
            "has_genetic_data": "yes", "match_method": "exact",
        })

    with pytest.raises(AssertionError):
        resolve_tip_names(coverage_path, ["Species_not_present"])
