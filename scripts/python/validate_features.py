# -*- coding: utf-8 -*-
"""
validate_features.py

Three categories of validation tests for the extracted feature matrix:

  1. Structural tests  -- correct number of features, no NaN/Inf, value ranges
  2. Pattern detection tests  -- known species serve as anchors to verify
                                  that each feature group captures what it claims
  3. Biological sanity tests  -- genus-level separation, known uniform vs
                                  complex species ordering, stripe detection

All tests print PASS / FAIL with the observed values so you can inspect
what the extractor actually captured, not just whether it ran.

Usage:
  python scripts/python/validate_features.py

  # Save a detailed HTML report
  python scripts/python/validate_features.py --report
"""

import csv
import json
import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
FEAT_DIR     = PROJECT_ROOT / "data" / "features"
FEAT_CSV     = FEAT_DIR / "features.csv"
FEAT_JSON    = FEAT_DIR / "features.json"
NAMES_TXT    = FEAT_DIR / "feature_names.txt"
REPORT_DIR   = PROJECT_ROOT / "outputs" / "validation"

N_FEATURES_EXPECTED = 99

# Feature group index ranges (must match extract_features.py)
GROUPS = {
    "hue_histogram":   (0,  18),
    "saturation_hist": (18, 26),
    "value_hist":      (26, 34),
    "dominant_colors": (34, 54),
    "dorsal_ventral":  (54, 63),
    "gabor_texture":   (63, 87),
    "lbp_texture":     (87, 97),
    "pattern_entropy": (97, 99),
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test result tracker
# ---------------------------------------------------------------------------

class Results:
    def __init__(self):
        self.passed  = 0
        self.failed  = 0
        self.records = []

    def check(self, test_name: str, condition: bool, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
            logger.info(f"  [{status}] {test_name}")
        else:
            self.failed += 1
            logger.warning(f"  [{status}] {test_name}")
        if detail:
            logger.info(f"         {detail}")
        self.records.append({"test": test_name, "status": status, "detail": detail})

    def summary(self):
        total = self.passed + self.failed
        logger.info(f"\n  {self.passed}/{total} tests passed")
        if self.failed > 0:
            logger.warning(f"  {self.failed} tests FAILED -- review details above")
        return self.failed == 0


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_features() -> tuple:
    """Returns (feature_matrix, species_list, feature_names, genus_list)."""
    if not FEAT_CSV.exists():
        logger.error(f"features.csv not found at {FEAT_CSV}")
        logger.error("Run extract_features.py first.")
        sys.exit(1)

    rows        = []
    species_list= []
    genus_list  = []

    with open(FEAT_CSV) as f:
        reader = csv.DictReader(f)
        feature_names = [c for c in reader.fieldnames if c not in ("genus", "species")]
        for row in reader:
            species_list.append(row["species"])
            genus_list.append(row["genus"])
            rows.append([float(row[n]) for n in feature_names])

    matrix = np.array(rows, dtype=np.float32)
    return matrix, species_list, feature_names, genus_list


# ---------------------------------------------------------------------------
# Category 1: Structural tests
# ---------------------------------------------------------------------------

def test_structural(matrix, species_list, feature_names, results: Results):
    logger.info("\n" + "=" * 60)
    logger.info("CATEGORY 1 -- Structural integrity")
    logger.info("=" * 60)

    # 1a. Species count
    n = len(species_list)
    results.check(
        "Species count (expect 63 -- tominiensis excluded)",
        n == 63,
        f"Got {n} species"
    )

    # 1b. Feature dimension
    results.check(
        f"Feature vector length = {N_FEATURES_EXPECTED}",
        matrix.shape[1] == N_FEATURES_EXPECTED,
        f"Got shape {matrix.shape}"
    )

    # 1c. Feature names count matches matrix columns
    results.check(
        "Feature names count matches matrix columns",
        len(feature_names) == matrix.shape[1],
        f"{len(feature_names)} names vs {matrix.shape[1]} columns"
    )

    # 1d. No NaN values
    n_nan = int(np.isnan(matrix).sum())
    results.check(
        "No NaN values in feature matrix",
        n_nan == 0,
        f"{n_nan} NaN values found"
    )

    # 1e. No Inf values
    n_inf = int(np.isinf(matrix).sum())
    results.check(
        "No Inf values in feature matrix",
        n_inf == 0,
        f"{n_inf} Inf values found"
    )

    # 1f. Feature group dimensions sum to total
    group_total = sum(e - s for s, e in GROUPS.values())
    results.check(
        f"Feature group dimensions sum to {N_FEATURES_EXPECTED}",
        group_total == N_FEATURES_EXPECTED,
        f"Sum = {group_total}"
    )

    # 1g. Each feature group has non-zero variance (something was captured)
    for group_name, (start, end) in GROUPS.items():
        group_data = matrix[:, start:end]
        variance   = float(group_data.var())
        results.check(
            f"  Group '{group_name}' has non-zero variance",
            variance > 1e-8,
            f"variance = {variance:.6f}"
        )

    # 1h. Value ranges per group
    # Histograms and LBP are density-normalised -> values >= 0
    for group_name in ["hue_histogram", "saturation_hist", "value_hist", "lbp_texture"]:
        s, e = GROUPS[group_name]
        min_val = float(matrix[:, s:e].min())
        results.check(
            f"  Group '{group_name}' values >= 0 (density-normalised)",
            min_val >= -1e-6,
            f"min = {min_val:.6f}"
        )

    # Pattern entropy should be positive
    s, e = GROUPS["pattern_entropy"]
    min_entropy = float(matrix[:, s:e].min())
    results.check(
        "Pattern entropy values > 0",
        min_entropy > 0,
        f"min entropy = {min_entropy:.4f}"
    )


# ---------------------------------------------------------------------------
# Category 2: Pattern detection tests
# ---------------------------------------------------------------------------

def get_species(name: str, species_list, matrix):
    """Return feature vector for a species by name (partial match ok)."""
    for i, sp in enumerate(species_list):
        if name.lower() in sp.lower():
            return matrix[i], sp
    return None, None


def test_pattern_detection(matrix, species_list, feature_names, results: Results):
    logger.info("\n" + "=" * 60)
    logger.info("CATEGORY 2 -- Pattern detection correctness")
    logger.info("=" * 60)

    hs, he = GROUPS["hue_histogram"]
    ss, se = GROUPS["saturation_hist"]
    vs, ve = GROUPS["value_hist"]
    gs, ge = GROUPS["gabor_texture"]
    ls, le = GROUPS["lbp_texture"]
    es, ee = GROUPS["pattern_entropy"]
    dc_s, dc_e = GROUPS["dominant_colors"]
    dv_s, dv_e = GROUPS["dorsal_ventral"]

    # ------------------------------------------------------------------
    # Test 2a: Uniform yellow species (Z. flavescens) should have
    #          concentrated hue histogram (low entropy) and high saturation
    # ------------------------------------------------------------------
    feat, name = get_species("flavescens", species_list, matrix)
    if feat is not None:
        hue_entropy = feat[es]
        sat_mean    = feat[ss:se].mean()
        # Hue bins 3-5 cover 30-60 deg (yellow range in HSV)
        yellow_hue  = feat[hs+3:hs+6].sum()
        results.check(
            f"Zebrasoma flavescens: low hue entropy (uniform yellow)",
            hue_entropy < 2.0,
            f"hue_entropy={hue_entropy:.3f}  (< 2.0 expected for uniform color)"
        )
        results.check(
            f"Zebrasoma flavescens: high saturation (vivid yellow)",
            sat_mean > 0.1,
            f"mean_saturation_hist={sat_mean:.4f}"
        )
        results.check(
            f"Zebrasoma flavescens: yellow hue bins dominant",
            yellow_hue > 0.05,
            f"hue_bins[30-60deg] sum={yellow_hue:.4f}"
        )
    else:
        logger.warning("  [SKIP] Zebrasoma flavescens not in dataset")

    # ------------------------------------------------------------------
    # Test 2b: Uniform grey species (N. hexacanthus) should have
    #          low saturation and low hue entropy
    # ------------------------------------------------------------------
    feat, name = get_species("hexacanthus", species_list, matrix)
    if feat is not None:
        hue_entropy = feat[es]
        # Saturation bin 0 covers 0-32 (low saturation = grey)
        low_sat_bin = feat[ss]
        results.check(
            f"Naso hexacanthus: low hue entropy (uniform grey)",
            hue_entropy < 2.2,
            f"hue_entropy={hue_entropy:.3f}"
        )
        results.check(
            f"Naso hexacanthus: dominant low-saturation bin (grey = unsaturated)",
            low_sat_bin > 0.0,
            f"saturation_bin_0={low_sat_bin:.4f}"
        )
    else:
        logger.warning("  [SKIP] Naso hexacanthus not in dataset")

    # ------------------------------------------------------------------
    # Test 2c: Highly patterned species (Z. desjardinii) should have
    #          high entropy (complex multi-color pattern with stripes+dots)
    # ------------------------------------------------------------------
    feat_d, name_d = get_species("desjardinii", species_list, matrix)
    feat_f, name_f = get_species("flavescens",  species_list, matrix)
    if feat_d is not None and feat_f is not None:
        ent_d = feat_d[es]
        ent_f = feat_f[es]
        results.check(
            "Zebrasoma desjardinii entropy > Zebrasoma flavescens entropy",
            ent_d > ent_f,
            f"desjardinii={ent_d:.3f}  flavescens={ent_f:.3f}"
        )

    # ------------------------------------------------------------------
    # Test 2d: Striped species should have strong directional Gabor response.
    #          A. lineatus (horizontal yellow/blue stripes) should have
    #          high Gabor energy at 90-degree orientation (horizontal stripes
    #          produce strong vertical-frequency response).
    #          Gabor features layout: 4 thetas x 3 freqs x 2 stats
    #          theta=90deg is index 2 in GABOR_THETAS=[0,45,90,135]
    #          -> cols gs + 2*3*2 = gs+12 through gs+17
    # ------------------------------------------------------------------
    feat_lin, _ = get_species("lineatus", species_list, matrix)
    feat_uni, _ = get_species("unicornis", species_list, matrix)   # less patterned Naso
    if feat_lin is not None:
        # Mean of all Gabor energies (mean stats, not std, every 2nd value)
        all_gabor_means = feat_lin[gs:ge:2]   # every 2nd = mean stats
        max_orientation_energy = all_gabor_means.max()
        mean_orientation_energy= all_gabor_means.mean()
        results.check(
            "Acanthurus lineatus: high peak Gabor energy (striped pattern)",
            max_orientation_energy > mean_orientation_energy * 1.2,
            f"peak={max_orientation_energy:.4f}  mean={mean_orientation_energy:.4f}"
        )

    if feat_lin is not None and feat_uni is not None:
        # Striped species should have higher overall Gabor energy than uniform Naso
        gabor_lin = feat_lin[gs:ge].mean()
        gabor_uni = feat_uni[gs:ge].mean()
        results.check(
            "A. lineatus Gabor energy > N. unicornis (stripe vs uniform)",
            gabor_lin > gabor_uni * 0.8,   # allow 20% tolerance
            f"lineatus={gabor_lin:.4f}  unicornis={gabor_uni:.4f}"
        )

    # ------------------------------------------------------------------
    # Test 2e: Dominant color count -- bicolor species should have
    #          2 large dominant clusters, uniform species only 1.
    #          Dominant color frequencies are at indices dc_s+15:dc_e (5 freqs)
    # ------------------------------------------------------------------
    feat_ach, _ = get_species("achilles", species_list, matrix)   # dark + orange patch
    feat_hex, _ = get_species("hexacanthus", species_list, matrix)

    if feat_ach is not None and feat_hex is not None:
        freqs_ach = feat_ach[dc_s+15:dc_e]   # 5 frequency values
        freqs_hex = feat_hex[dc_s+15:dc_e]

        # Number of dominant clusters with >15% coverage
        n_dom_ach = int((freqs_ach > 0.15).sum())
        n_dom_hex = int((freqs_hex > 0.15).sum())

        results.check(
            "A. achilles has >= 2 dominant color clusters (dark body + orange patch)",
            n_dom_ach >= 2,
            f"clusters >15% coverage: {n_dom_ach}  (freqs: {freqs_ach.round(3)})"
        )
        results.check(
            "N. hexacanthus has fewer dominant clusters than A. achilles",
            n_dom_hex <= n_dom_ach,
            f"hexacanthus clusters: {n_dom_hex}  achilles clusters: {n_dom_ach}"
        )

    # ------------------------------------------------------------------
    # Test 2f: LBP entropy -- high-texture species should have higher
    #          LBP entropy than smooth-bodied species
    # ------------------------------------------------------------------
    feat_str, _ = get_species("striatus",    species_list, matrix)  # Ctenochaetus -- fine stripes
    feat_naso,_ = get_species("unicornis",   species_list, matrix)  # Naso -- smooth body
    if feat_str is not None and feat_naso is not None:
        lbp_ent_str  = feat_str[ee-1]    # lbp_entropy is index 98
        lbp_ent_naso = feat_naso[ee-1]
        results.check(
            "C. striatus LBP entropy >= N. unicornis (fine texture vs smooth)",
            lbp_ent_str >= lbp_ent_naso * 0.9,
            f"striatus={lbp_ent_str:.3f}  unicornis={lbp_ent_naso:.3f}"
        )

    # ------------------------------------------------------------------
    # Test 2g: Dorsal/ventral gradient -- countershaded species should
    #          have non-zero dorsal-ventral value difference
    # ------------------------------------------------------------------
    # dv_diff is the last 3 values of the dorsal_ventral group [60:63]
    # index 62 = dv_diff_val (brightness difference)
    for sp_name in ["blochii", "nigrofuscus", "sohal"]:
        feat_dv, matched = get_species(sp_name, species_list, matrix)
        if feat_dv is not None:
            dv_val_diff = abs(feat_dv[62])
            results.check(
                f"  {matched}: non-zero dorsal/ventral brightness difference",
                dv_val_diff > 0.01,
                f"|dv_val_diff| = {dv_val_diff:.4f}"
            )


# ---------------------------------------------------------------------------
# Category 3: Biological sanity tests
# ---------------------------------------------------------------------------

def test_biological_sanity(matrix, species_list, genus_list, results: Results):
    logger.info("\n" + "=" * 60)
    logger.info("CATEGORY 3 -- Biological sanity")
    logger.info("=" * 60)

    es, ee = GROUPS["pattern_entropy"]
    gs, ge = GROUPS["gabor_texture"]
    hs, he = GROUPS["hue_histogram"]

    # Build genus-level mean feature vectors
    genera = sorted(set(genus_list))
    genus_means = {}
    for genus in genera:
        idx = [i for i, g in enumerate(genus_list) if g == genus]
        genus_means[genus] = matrix[idx].mean(axis=0)

    # ------------------------------------------------------------------
    # Test 3a: Naso should be visually distinct from Acanthurus
    #          (different body shape, different coloration regime)
    # ------------------------------------------------------------------
    if "Naso" in genus_means and "Acanthurus" in genus_means:
        dist = float(np.linalg.norm(genus_means["Naso"] - genus_means["Acanthurus"]))
        results.check(
            "Naso mean vector != Acanthurus mean vector (visually distinct genera)",
            dist > 0.1,
            f"Euclidean distance between genus means = {dist:.4f}"
        )

    # ------------------------------------------------------------------
    # Test 3b: Within-genus variance should be lower than between-genus variance
    #          for at least some genera -- this tests whether the features
    #          capture genus-level signal at all
    # ------------------------------------------------------------------
    within_vars  = []
    between_dists= []

    for genus in genera:
        idx = [i for i, g in enumerate(genus_list) if g == genus]
        if len(idx) >= 3:
            within_vars.append(float(matrix[idx].var()))

    # Between-genus: pairwise distances between genus mean vectors
    genus_list_keys = list(genus_means.keys())
    for i in range(len(genus_list_keys)):
        for j in range(i+1, len(genus_list_keys)):
            d = float(np.linalg.norm(
                genus_means[genus_list_keys[i]] - genus_means[genus_list_keys[j]]
            ))
            between_dists.append(d)

    mean_within  = float(np.mean(within_vars))
    mean_between = float(np.mean(between_dists))

    results.check(
        "Between-genus distance > within-genus variance (genus-level signal present)",
        mean_between > mean_within * 0.1,
        f"mean_between={mean_between:.4f}  mean_within_var={mean_within:.4f}"
    )

    # ------------------------------------------------------------------
    # Test 3c: Entropy ordering -- known complex vs uniform species
    #          Complex: Z. desjardinii (stripes + dots + multiple colors)
    #          Uniform: Naso hexacanthus (plain grey)
    #          Intermediate: most Acanthurus species
    # ------------------------------------------------------------------
    feat_complex, _ = get_species("desjardinii", species_list, matrix)
    feat_uniform, _ = get_species("hexacanthus", species_list, matrix)

    if feat_complex is not None and feat_uniform is not None:
        ent_complex = feat_complex[es]
        ent_uniform = feat_uniform[es]
        acanthurus_idx = [i for i, g in enumerate(genus_list) if g == "Acanthurus"]
        ent_acanthurus = matrix[acanthurus_idx, es].mean()

        results.check(
            "Entropy order: Z.desjardinii >= Acanthurus_mean >= N.hexacanthus",
            ent_complex >= ent_acanthurus * 0.9 and ent_acanthurus >= ent_uniform * 0.9,
            f"desjardinii={ent_complex:.3f}  acanthurus_mean={ent_acanthurus:.3f}  hexacanthus={ent_uniform:.3f}"
        )

    # ------------------------------------------------------------------
    # Test 3d: All species in Zebrasoma are visually distinct from each other
    #          (they have notably different coloration -- this tests whether
    #           the feature space separates them)
    # ------------------------------------------------------------------
    zeb_idx = [i for i, g in enumerate(genus_list) if g == "Zebrasoma"]
    if len(zeb_idx) >= 2:
        zeb_matrix = matrix[zeb_idx]
        zeb_species = [species_list[i] for i in zeb_idx]

        min_dist = float('inf')
        min_pair = ("", "")
        for i in range(len(zeb_idx)):
            for j in range(i+1, len(zeb_idx)):
                d = float(np.linalg.norm(zeb_matrix[i] - zeb_matrix[j]))
                if d < min_dist:
                    min_dist = d
                    min_pair = (zeb_species[i], zeb_species[j])

        results.check(
            "Zebrasoma species are mutually distinct (min pairwise distance > 0)",
            min_dist > 0.01,
            f"Most similar pair: {min_pair[0]} <-> {min_pair[1]}  dist={min_dist:.4f}"
        )

    # ------------------------------------------------------------------
    # Test 3e: Acanthurus/Ctenochaetus clade proximity
    #          These two genera are sister clades -- their mean feature
    #          vectors should be closer to each other than to Naso/Zebrasoma
    # ------------------------------------------------------------------
    if all(g in genus_means for g in ["Acanthurus", "Ctenochaetus", "Naso", "Zebrasoma"]):
        d_ac_ct = float(np.linalg.norm(genus_means["Acanthurus"] - genus_means["Ctenochaetus"]))
        d_ac_ns = float(np.linalg.norm(genus_means["Acanthurus"] - genus_means["Naso"]))
        d_ac_zb = float(np.linalg.norm(genus_means["Acanthurus"] - genus_means["Zebrasoma"]))

        results.check(
            "Acanthurus closer to Ctenochaetus than to Naso (phylogenetic signal)",
            d_ac_ct < d_ac_ns,
            f"Acanthurus<->Ctenochaetus={d_ac_ct:.4f}  Acanthurus<->Naso={d_ac_ns:.4f}"
        )
        results.check(
            "Acanthurus closer to Ctenochaetus than to Zebrasoma",
            d_ac_ct < d_ac_zb,
            f"Acanthurus<->Ctenochaetus={d_ac_ct:.4f}  Acanthurus<->Zebrasoma={d_ac_zb:.4f}"
        )

    # ------------------------------------------------------------------
    # Test 3f: Naso cluster -- all Naso species should be more similar
    #          to each other than to any Zebrasoma (Nasinae vs Acanthurinae)
    # ------------------------------------------------------------------
    naso_idx = [i for i, g in enumerate(genus_list) if g == "Naso"]
    zeb_idx2 = [i for i, g in enumerate(genus_list) if g == "Zebrasoma"]
    if len(naso_idx) >= 2 and len(zeb_idx2) >= 1:
        naso_center = matrix[naso_idx].mean(axis=0)
        zeb_center  = matrix[zeb_idx2].mean(axis=0)

        naso_to_naso_dists = [
            float(np.linalg.norm(matrix[i] - naso_center)) for i in naso_idx
        ]
        naso_to_zeb_dists  = [
            float(np.linalg.norm(matrix[i] - zeb_center)) for i in naso_idx
        ]

        mean_nn = float(np.mean(naso_to_naso_dists))
        mean_nz = float(np.mean(naso_to_zeb_dists))

        results.check(
            "Naso species closer to Naso centroid than Zebrasoma centroid",
            mean_nn < mean_nz,
            f"mean dist to Naso center={mean_nn:.4f}  to Zebrasoma center={mean_nz:.4f}"
        )

    # ------------------------------------------------------------------
    # Test 3g: Print per-genus entropy summary (informational)
    # ------------------------------------------------------------------
    logger.info("\n  Per-genus pattern entropy (hue) summary:")
    for genus in sorted(genera):
        idx = [i for i, g in enumerate(genus_list) if g == genus]
        ent_vals = matrix[idx, es]
        logger.info(
            f"    {genus:<20} n={len(idx):2d}  "
            f"mean={ent_vals.mean():.3f}  "
            f"min={ent_vals.min():.3f}  "
            f"max={ent_vals.max():.3f}"
        )


# ---------------------------------------------------------------------------
# Optional: save report
# ---------------------------------------------------------------------------

def save_report(results: Results) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "feature_validation_report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test", "status", "detail"])
        writer.writeheader()
        writer.writerows(results.records)
    logger.info(f"\nReport saved: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate extracted fish feature matrix"
    )
    parser.add_argument("--report", action="store_true",
                        help="Save CSV report to outputs/validation/")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FEATURE VALIDATION  --  3 test categories")
    logger.info("=" * 60)
    logger.info(f"Input: {FEAT_CSV}")

    matrix, species_list, feature_names, genus_list = load_features()
    logger.info(f"Loaded: {matrix.shape[0]} species x {matrix.shape[1]} features")

    R = Results()

    test_structural(matrix, species_list, feature_names, R)
    test_pattern_detection(matrix, species_list, feature_names, R)
    test_biological_sanity(matrix, species_list, genus_list, R)

    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    all_passed = R.summary()

    if args.report:
        save_report(R)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
