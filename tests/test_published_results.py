"""Integrity checks on the published result files themselves.

Different in kind from the rest of the suite: every other test exercises
``src/`` logic against synthetic inputs, while these read the real tracked
outputs under ``reports/`` and ``outputs/`` and assert that the numbers
quoted in README.md and METHODS.md actually follow from them.

Two things this catches that unit tests cannot:

1. **Cross-file drift.** The extraction state, the event log, the review
   feedback, the per-image features, and the per-species aggregate all encode
   overlapping facts. If one is regenerated without the others, the counts stop
   reconciling - and that is exactly the kind of silent inconsistency this
   project was rebuilt to avoid.
2. **Cross-language verification.** The Mantel statistics and the
   Benjamini-Hochberg correction were computed in R (``r/phase4_kmult.R``).
   Here they are recomputed independently in Python from the same distance
   matrices, so the published numbers are checked rather than trusted.

Skipped automatically if the result files are absent, so a fresh clone that
has not run the pipeline still gets a green suite.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "data/fish_extraction_state.json",
    "reports/fish_extraction_log.csv",
    "reports/pattern_features.csv",
    "reports/species_features.csv",
    "outputs/color_distance_matrix.csv",
    "outputs/patristic_distance_matrix.csv",
    "outputs/phase4/mantel_results.csv",
]

pytestmark = pytest.mark.skipif(
    not all((ROOT / p).exists() for p in REQUIRED),
    reason="published result files not present in this checkout",
)

# The Kmult results come from r/phase4_kmult.R, which needs R + geomorph and so
# is regenerated on Colab rather than locally. After a change that invalidates
# them (e.g. the v6.2.0 stripe recalibration) they are deliberately absent until
# that re-run happens, rather than left stale alongside fresh Mantel results -
# a stale file that still looks current is precisely the cross-file
# inconsistency this module exists to catch.
needs_kmult = pytest.mark.skipif(
    not (ROOT / "outputs/phase4/kmult_results.csv").exists(),
    reason="Kmult results pending regeneration in R (see notebooks/Followups.ipynb)",
)

# compare.physignal.z's result was console-only until v6.7.4, so this file does
# not exist in checkouts predating the next Phase 4 run. Skipping until then is
# the point: the alternative is asserting nothing at all about those numbers.
needs_comparison = pytest.mark.skipif(
    not (ROOT / "outputs/phase4/comparison_results.csv").exists(),
    reason="cross-dimension comparison export pending an R re-run "
           "(see notebooks/Phase4_Comparison_Export.ipynb)",
)

DIMENSIONS = ["color", "stripe", "spot"]


def _read_csv(relative_path: str) -> list[dict]:
    with open(ROOT / relative_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_matrix(relative_path: str) -> tuple[list[str], np.ndarray]:
    """Reads a species x species distance matrix CSV.

    Args:
        relative_path: Path relative to the repository root.

    Returns:
        The species labels in row order, and the matrix itself.
    """
    with open(ROOT / relative_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    species = [r[0] for r in rows[1:]]
    matrix = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    return species, matrix


def _benjamini_hochberg(pvalues: list[float]) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR correction.

    Independent reimplementation of R's ``p.adjust(method = "BH")``, so the
    published corrected p-values are verified rather than copied.

    Args:
        pvalues: Raw p-values.

    Returns:
        Corrected p-values, in the same order as the input.
    """
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    scaled = p[order] * n / (np.arange(n) + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    corrected = np.empty(n)
    corrected[order] = np.minimum(scaled, 1.0)
    return corrected


# --------------------------------------------------------------------------
# Extraction chain
# --------------------------------------------------------------------------


def test_extraction_state_totals_match_the_reported_counts():
    state = json.loads(
        (ROOT / "data/fish_extraction_state.json").read_text(encoding="utf-8")
    )
    statuses = [v["status"] for v in state.values()]

    assert len(state) == 1460
    assert statuses.count("accepted") == 856
    assert statuses.count("excluded") == 604
    assert "flagged" not in statuses


def test_event_log_and_state_agree_on_which_images_were_accepted():
    # The log is append-only across review rounds (more rows than images), so
    # only the per-key accepted set is comparable - not raw row counts.
    state = json.loads(
        (ROOT / "data/fish_extraction_state.json").read_text(encoding="utf-8")
    )
    log = _read_csv("reports/fish_extraction_log.csv")

    from_state = {k for k, v in state.items() if v["status"] == "accepted"}
    from_log = {r["image_key"] for r in log if r["status"] == "accepted"}

    assert from_state == from_log


def test_review_rounds_reconcile_to_the_excluded_total():
    # 605 exclusion decisions across three non-overlapping rounds, minus the
    # one image later restored by human override, gives the reported 604.
    decisions: set[str] = set()
    for round_number in (1, 2, 3):
        path = ROOT / f"reports/fish_extraction_review_feedback_round{round_number}.json"
        decisions |= set(json.loads(path.read_text(encoding="utf-8"))["excluded"])

    state = json.loads(
        (ROOT / "data/fish_extraction_state.json").read_text(encoding="utf-8")
    )
    accepted = {k for k, v in state.items() if v["status"] == "accepted"}
    restored = decisions & accepted

    assert len(decisions) == 605
    assert len(restored) == 1
    assert len(decisions) - len(restored) == 604


def test_pattern_features_covers_every_accepted_image_except_the_validation_exclusions():
    state = json.loads(
        (ROOT / "data/fish_extraction_state.json").read_text(encoding="utf-8")
    )
    accepted = {
        k.rsplit(".", 1)[0] for k, v in state.items() if v["status"] == "accepted"
    }
    features = _read_csv("reports/pattern_features.csv")
    keys = {r["image_key"] for r in features}

    # Three images were removed at validation as invalid inputs regardless of
    # pattern: a fin-only crop, a blown-out photo, a dried museum specimen.
    assert len(features) == 853 == len(accepted) - 3
    assert not keys - accepted, "pattern rows exist for non-accepted images"


# --------------------------------------------------------------------------
# Detector validation
# --------------------------------------------------------------------------

# The detector table published in README.md and METHODS.md, as
# dimension -> (agreement, precision, recall, f1, true positives).
PUBLISHED_DETECTOR_METRICS = {
    "is_solid": (0.703, 0.712, 0.849, 0.775, 79),
    "stripe_present": (0.774, 0.400, 0.500, 0.444, 14),
    "spot_present": (0.800, 0.105, 0.125, 0.114, 2),
}


def _load_labels() -> dict[str, dict]:
    return json.loads(
        (ROOT / "reports/pattern_validation_labels.json").read_text(encoding="utf-8")
    )["labels"]


def _detector_stats() -> dict[str, tuple]:
    """Recomputes each detector's confusion metrics from the raw inputs.

    Deliberately reads the hand labels and ``pattern_features.csv`` rather
    than the comparison report, so the report itself can be checked against
    this instead of being trusted as the source.

    Returns:
        dimension -> (agreement, precision, recall, f1, true positives).
    """
    labels = _load_labels()
    features = {r["image_key"]: r for r in _read_csv("reports/pattern_features.csv")}

    stats = {}
    for dimension in PUBLISHED_DETECTOR_METRICS:
        tp = fp = fn = tn = 0
        for image_key, label in labels.items():
            row = features.get(image_key)
            if row is None:
                continue
            extracted = row[dimension] == "True"
            manual = bool(label[dimension])
            tp += extracted and manual
            fp += extracted and not manual
            fn += (not extracted) and manual
            tn += (not extracted) and not manual
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        stats[dimension] = (
            (tp + tn) / (tp + fp + fn + tn),
            precision,
            recall,
            2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            tp,
        )
    return stats


def test_published_detector_metrics_follow_from_the_hand_labels():
    stats = _detector_stats()
    names = ("agreement", "precision", "recall", "f1")

    for dimension, published in PUBLISHED_DETECTOR_METRICS.items():
        *rates, tp = stats[dimension]
        assert tp == published[4], (
            f"{dimension} true positives: {tp}, documentation claims {published[4]}"
        )
        for value, claimed, name in zip(rates, published, names):
            assert abs(value - claimed) < 0.001, (
                f"{dimension} {name}: {value:.3f}, documentation claims {claimed}"
            )


def test_validation_report_is_not_stale_relative_to_the_features_it_summarises():
    """The comparison report is derived, so it can silently outlive its inputs.

    It did exactly that: it survived the v6.2.0 stripe recalibration without
    being regenerated, so it went on describing thresholds the code no longer
    used - 14.3% stripe recall against the 50.0% that README, METHODS and the
    figures all reported - while every other file moved. Nothing caught it
    because nothing compared the report to what it claims to summarise.
    """
    labels = _load_labels()
    features = {r["image_key"]: r for r in _read_csv("reports/pattern_features.csv")}

    drifted = []
    for row in _read_csv("reports/pattern_validation_report.csv"):
        key, dimension = row["image_key"], row["dimension"]
        current = features[key][dimension] == "True"
        manual = bool(labels[key][dimension])

        if (row["extracted"] == "True") != current:
            drifted.append(f"{key}/{dimension}")
        assert (row["manual"] == "True") == manual, f"{key}/{dimension} label drifted"
        assert (row["agree"] == "True") == (manual == current), (
            f"{key}/{dimension} agree column disagrees with its own two columns"
        )

    assert not drifted, (
        f"{len(drifted)} report row(s) disagree with pattern_features.csv - "
        "regenerate via pattern_extractor.validation.compare_to_features"
    )


def test_stripe_recalibration_baseline_still_supports_the_claim_made_about_it():
    """Guards the figure caption: "recall tripled at identical precision"."""
    baseline = {r["metric"]: float(r["value"])
                for r in _read_csv("reports/stripe_recalibration_baseline.csv")}
    _, precision, recall, _, _ = _detector_stats()["stripe_present"]

    assert abs(baseline["precision"] - precision) < 0.001, "precision was not held fixed"
    assert recall > 3 * baseline["recall"] - 0.01, "recall no longer roughly tripled"


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_species_aggregate_matches_the_per_image_rows_it_came_from():
    aggregates = _read_csv("reports/species_features.csv")
    features = _read_csv("reports/pattern_features.csv")
    analysed = {r["species"] for r in aggregates}

    # pattern_features spans all 64 species; only the phylogeny-matched ones
    # are aggregated, and reference photos are excluded by default.
    contributing = [
        r
        for r in features
        if r["is_reference"] == "False" and r["image_key"].split("/")[1] in analysed
    ]

    assert len(aggregates) == 49
    assert sum(int(r["n_images"]) for r in aggregates) == len(contributing) == 648


def test_aggregate_proportions_are_within_the_unit_interval():
    aggregates = _read_csv("reports/species_features.csv")

    for row in aggregates:
        for field in ("prop_solid", "prop_striped", "prop_spotted"):
            assert 0.0 <= float(row[field]) <= 1.0, f"{row['species']}.{field}"


def test_sensitivity_set_drops_exactly_the_species_below_the_cutoff():
    primary = _read_csv("reports/species_features.csv")
    reduced = _read_csv("reports/species_features_min5.csv")

    primary_species = {r["species"] for r in primary}
    reduced_species = {r["species"] for r in reduced}
    dropped = primary_species - reduced_species

    assert len(reduced) == 47
    assert dropped == {"Acanthurus_triostegus", "Naso_tuberosus"}
    for row in primary:
        below_cutoff = int(row["n_images"]) < 5
        assert below_cutoff == (row["species"] in dropped), row["species"]


# --------------------------------------------------------------------------
# Distance matrices
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_pattern_distance_matrix_is_a_valid_distance_matrix(dimension):
    species, matrix = _read_matrix(f"outputs/{dimension}_distance_matrix.csv")
    aggregates = _read_csv("reports/species_features.csv")

    assert matrix.shape == (49, 49)
    assert np.allclose(matrix, matrix.T, atol=1e-9), "not symmetric"
    assert np.allclose(np.diag(matrix), 0.0, atol=1e-9), "non-zero diagonal"
    assert (matrix >= -1e-12).all(), "negative distance"
    assert species == [r["species"] for r in aggregates], "species order drift"


def test_patristic_matrix_aligns_with_the_pattern_matrices():
    patristic_species, patristic = _read_matrix("outputs/patristic_distance_matrix.csv")
    pattern_species, _ = _read_matrix("outputs/color_distance_matrix.csv")

    assert patristic.shape == (49, 49)
    assert np.allclose(patristic, patristic.T, atol=1e-9)
    assert np.allclose(np.diag(patristic), 0.0, atol=1e-9)
    assert (patristic[~np.eye(49, dtype=bool)] > 0).all(), "zero off-diagonal distance"
    # geomorph matches by name, but the Mantel step indexes positionally.
    assert patristic_species == pattern_species


# --------------------------------------------------------------------------
# Published statistics, recomputed independently of R
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_published_mantel_r_is_reproducible_in_python(dimension):
    _, pattern = _read_matrix(f"outputs/{dimension}_distance_matrix.csv")
    _, patristic = _read_matrix("outputs/patristic_distance_matrix.csv")
    published = {
        r["dimension"]: float(r["observed_r"])
        for r in _read_csv("outputs/phase4/mantel_results.csv")
    }

    lower = np.tril_indices(pattern.shape[0], k=-1)
    recomputed = float(np.corrcoef(pattern[lower], patristic[lower])[0, 1])

    assert recomputed == pytest.approx(published[dimension], abs=1e-6)


@pytest.mark.parametrize("filename", ["mantel_results.csv"])
def test_published_bh_correction_is_reproducible_in_python(filename):
    rows = _read_csv(f"outputs/phase4/{filename}")

    recomputed = _benjamini_hochberg([float(r["raw_p"]) for r in rows])
    published = [float(r["bh_corrected_p"]) for r in rows]

    assert recomputed == pytest.approx(published, abs=1e-9)


def test_secondary_test_still_finds_colour_significant_and_stripe_null():
    """Mantel-only headline check, valid even while Kmult is pending."""
    mantel = {r["dimension"]: float(r["bh_corrected_p"]) for r in
              _read_csv("outputs/phase4/mantel_results.csv")}

    assert mantel["color"] < 0.05, "colour lost significance under the secondary test"
    assert mantel["stripe"] >= 0.05, "stripe is no longer null"


@needs_kmult
def test_headline_verdicts_are_what_the_documentation_claims():
    kmult = {r["dimension"]: float(r["bh_corrected_p"]) for r in
             _read_csv("outputs/phase4/kmult_results.csv")}
    mantel = {r["dimension"]: float(r["bh_corrected_p"]) for r in
              _read_csv("outputs/phase4/mantel_results.csv")}

    # Colour is the only dimension significant under the primary test.
    assert kmult["color"] < 0.05
    assert kmult["spot"] >= 0.05
    assert kmult["stripe"] >= 0.05
    # And it is corroborated by the secondary test.
    assert mantel["color"] < 0.05


@needs_kmult
def test_sensitivity_run_leaves_every_primary_verdict_unchanged():
    rows = _read_csv("outputs/phase4/sensitivity_comparison.csv")

    for row in rows:
        primary = float(row["kmult_p_primary"]) < 0.05
        sensitivity = float(row["kmult_p_sensitivity"]) < 0.05
        assert primary == sensitivity, f"{row['dimension']} primary verdict flipped"

    # The one documented instability is spot's *secondary* result.
    spot = next(r for r in rows if r["dimension"] == "spot")
    assert spot["mantel_verdict_stable"].strip().upper() == "FALSE"


@needs_comparison
def test_readme_cross_dimension_p_values_come_from_the_comparison_output():
    """The gap that let a superseded paragraph survive five versions.

    README quoted `compare.physignal.z` p-values that existed only in R console
    output. With no file to check against, a pre-recalibration paragraph could
    sit directly beneath the corrected one - each stating different values for
    the same two comparisons - and nothing could tell them apart.
    """
    values = {round(float(r["value"]), 3)
              for r in _read_csv("outputs/phase4/comparison_results.csv")}

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("Cross-dimension comparison")
    section = readme[start:readme.index("### How to read these numbers", start)]

    quoted = {round(float(v), 3) for v in re.findall(r"\*p\*=([0-9.]+)", section)}
    # Values explicitly marked "was X" are labelled superseded on purpose.
    superseded = {round(float(v), 3) for v in re.findall(r"was ([0-9.]+)", section)}

    missing = sorted((quoted - superseded) - values)
    assert not missing, (
        f"README quotes cross-dimension p-value(s) {missing} absent from "
        "outputs/phase4/comparison_results.csv"
    )


# --------------------------------------------------------------------------
# Phase 4 inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_kmult_feature_matrix_is_standardized_and_well_conditioned(dimension):
    _, matrix = _read_matrix(f"outputs/phase4/{dimension}_kmult_features.csv")

    assert np.isfinite(matrix).all()
    # Written to 6 decimal places, so the tolerance reflects file precision.
    assert np.allclose(matrix.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(matrix.std(axis=0), 1.0, atol=1e-6)
    assert np.linalg.cond(matrix) < 1e10, "ill-conditioned input to physignal.z"


def test_pruned_tree_tips_match_the_analysis_species_exactly():
    newick = (ROOT / "outputs/phase4/pruned_tree.nwk").read_text(encoding="utf-8")
    aggregates = _read_csv("reports/species_features.csv")

    tips = set()
    for token in newick.replace("(", ",").replace(")", ",").split(","):
        name = token.split(":")[0].strip()
        if name and not name.replace(".", "").replace("-", "").isdigit():
            tips.add(name)

    assert tips == {r["species"] for r in aggregates}


def test_phylogeny_coverage_table_still_reports_fifty_matched_species():
    coverage = _read_csv("data/phylogeny/species_coverage.csv")
    matched = [
        r for r in coverage if r["has_genetic_data"].strip().lower() in ("yes", "true")
    ]
    synonyms = [r for r in matched if "synonym" in r["match_method"].lower()]

    assert len(coverage) == 64
    assert len(matched) == 50
    # Exactly one match depends on the NCBI synonym table (Zebrasoma veliferum).
    assert len(synonyms) == 1
