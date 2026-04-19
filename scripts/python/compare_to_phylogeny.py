"""
compare_to_phylogeny.py

Statistical comparison between visual feature distance matrices and the
molecular phylogenetic distance matrix. Reads the aligned CSV outputs
produced by build_distance_matrix.py.

Analyses performed
------------------
  1. Mantel test (Pearson + Spearman, configurable permutation count)
     Run twice: once on the PCA-based visual matrix, once on the
     feature-selected matrix (dorsal_ventral + hue_hist + entropy + sat_hist).
     Reports r, rho, p-values, and a power estimate.

  2. Robinson-Foulds distance
     Compares the UPGMA visual dendrogram topology against the molecular tree
     using dendropy with a shared TaxonNamespace to avoid namespace errors.

  3. Visual similarity dendrogram (UPGMA) with abbreviated, genus-coloured
     leaf labels. Colouring uses the dendrogram ivl (leaf order) array as the
     authoritative label sequence rather than ax.get_yticklabels(), which is
     unreliable in the Agg backend.

  4. Tanglegram comparing the visual dendrogram to the molecular tree, with
     abbreviated genus-coloured leaf labels using the same ivl-based approach.

  5. Per-feature Mantel correlations across all 99 features.

Outputs
-------
  outputs/phylogenetic_analysis/mantel_test_results.json
  outputs/phylogenetic_analysis/mantel_permutation_distribution_pca.png
  outputs/phylogenetic_analysis/mantel_permutation_distribution_featureselected.png
  outputs/phylogenetic_analysis/visual_dendrogram.png
  outputs/phylogenetic_analysis/tanglegram.png
  outputs/phylogenetic_analysis/feature_mantel_correlations.csv
  outputs/phylogenetic_analysis/feature_mantel_correlations.png
  outputs/phylogenetic_analysis/analysis_summary.json

Usage
-----
  python scripts/python/compare_to_phylogeny.py
  python scripts/python/compare_to_phylogeny.py --n-permutations 99999
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_DIR          = PROJECT_ROOT / "data"
OUT_DIR           = PROJECT_ROOT / "outputs"
DIST_DIR          = OUT_DIR / "distance_matrices"
PHYL_DIR          = OUT_DIR / "phylogenetic_analysis"
TREE_FILE         = DATA_DIR / "phylogeny" / "Acanthuridae_timetree.tre"
FEATURES_CSV      = DATA_DIR / "features" / "features.csv"
VIS_ALIGNED_CSV   = DIST_DIR / "visual_distance_matrix_aligned.csv"
VISFS_ALIGNED_CSV = DIST_DIR / "visual_distance_matrix_featureselected_aligned.csv"
PAT_ALIGNED_CSV   = DIST_DIR / "patristic_distance_matrix_aligned.csv"

FEATURE_GROUP_MAP = (
    ["hue_hist"]        * 18 +
    ["sat_hist"]        * 8  +
    ["val_hist"]        * 8  +
    ["dom_colours"]     * 20 +
    ["dorsal_ventral"]  * 9  +
    ["gabor"]           * 24 +
    ["lbp"]             * 10 +
    ["entropy"]         * 2
)

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
# Styling
# ---------------------------------------------------------------------------

BG       = "#0f1117"
PANEL_BG = "#1a1d27"
TEXT_COL = "#e8e8f0"
SUBTLE   = "#666677"
BRANCH   = "#888899"

GENUS_COLOURS = {
    "Acanthurus"   : "#4fc3f7",
    "Ctenochaetus" : "#81c784",
    "Naso"         : "#ffb74d",
    "Paracanthurus": "#f06292",
    "Prionurus"    : "#ce93d8",
    "Zebrasoma"    : "#ff8a65",
}


# ===========================================================================
# Label utilities
# ===========================================================================

def clean_label(label):
    """
    Convert the internal storage key to a readable abbreviated display label.

    Examples:
      "Acanthurus Acanthurus lineatus"     -> "A. lineatus"
      "Acanthurus Acanthurus_pyroferus"    -> "A. pyroferus"
      "Paracanthurus Paracanthurus_hepatus"-> "P. hepatus"
    """
    parts = label.replace("_", " ").split()
    if len(parts) >= 3 and parts[0] == parts[1]:
        return parts[0][0] + ". " + " ".join(parts[2:])
    elif len(parts) >= 2:
        return parts[0][0] + ". " + " ".join(parts[1:])
    return label


def genus_of(label):
    return label.split()[0]


def genus_colour(label):
    return GENUS_COLOURS.get(genus_of(label), TEXT_COL)


def apply_leaf_colours_from_ivl(ax, ivl, colour_map, fontsize=8, axis="y"):
    """
    Apply genus colours to dendrogram leaf labels using the ivl (leaf-order)
    array returned by scipy's dendrogram() as the authoritative label sequence.

    This approach is reliable in the Agg backend because it rebuilds the tick
    labels from scratch with set_yticklabels / set_xticklabels rather than
    attempting to recolour existing tick objects after tight_layout, which
    discards colour assignments in non-interactive mode.

    Parameters
    ----------
    ax        : matplotlib Axes
    ivl       : list of str, leaf labels in rendered order (from ddata["ivl"])
    colour_map: dict mapping display label -> colour string
    fontsize  : int
    axis      : "y" or "x"
    """
    colours = [colour_map.get(lbl, TEXT_COL) for lbl in ivl]
    if axis == "y":
        ax.set_yticklabels(ivl, fontsize=fontsize)
        fig = ax.get_figure()
        fig.canvas.draw()
        for tick_lbl, col in zip(ax.get_yticklabels(), colours):
            tick_lbl.set_color(col)
    else:
        ax.set_xticklabels(ivl, fontsize=fontsize, rotation=90)
        fig = ax.get_figure()
        fig.canvas.draw()
        for tick_lbl, col in zip(ax.get_xticklabels(), colours):
            tick_lbl.set_color(col)


# ===========================================================================
# 1. Mantel test (Pearson + Spearman)
# ===========================================================================

def mantel_test(vis_mat, pat_mat, n_permutations=9999, seed=42):
    """
    Mantel test with Pearson and Spearman statistics.

    Extracts upper-triangular elements, computes observed statistics, then
    permutes rows/columns of the visual matrix n_permutations times to
    estimate one-tailed p-values. Also reports the minimum detectable r at
    alpha=0.05, power=0.80 given the number of pairwise distances.
    """
    rng = np.random.RandomState(seed)
    n   = vis_mat.shape[0]
    idx = np.triu_indices(n, k=1)

    v_vec = vis_mat[idx]
    p_vec = pat_mat[idx]

    r_obs,   _ = pearsonr(v_vec, p_vec)
    rho_obs, _ = spearmanr(v_vec, p_vec)

    permuted_r   = np.empty(n_permutations)
    permuted_rho = np.empty(n_permutations)

    for i in range(n_permutations):
        perm             = rng.permutation(n)
        v_perm           = vis_mat[np.ix_(perm, perm)][idx]
        permuted_r[i],   _ = pearsonr(v_perm, p_vec)
        permuted_rho[i], _ = spearmanr(v_perm, p_vec)

    p_pearson  = float((permuted_r   >= r_obs  ).sum() + 1) / (n_permutations + 1)
    p_spearman = float((permuted_rho >= rho_obs).sum() + 1) / (n_permutations + 1)
    n_pairs    = len(v_vec)
    min_r      = (1.645 + 0.842) / math.sqrt(n_pairs - 3)

    return {
        "r_obs"            : float(r_obs),
        "rho_obs"          : float(rho_obs),
        "p_pearson"        : p_pearson,
        "p_spearman"       : p_spearman,
        "permuted_r"       : permuted_r,
        "permuted_rho"     : permuted_rho,
        "n_pairs"          : n_pairs,
        "min_detectable_r" : float(min_r),
        "n_species"        : n,
    }


def plot_mantel_distribution(results, title, out_path):
    """Save a permutation distribution histogram with observed r and minimum
    detectable r annotated."""
    r_obs  = results["r_obs"]
    p_val  = results["p_pearson"]
    perm_r = results["permuted_r"]
    min_r  = results["min_detectable_r"]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(PANEL_BG)
    ax.hist(perm_r, bins=60, color=SUBTLE, edgecolor="none", alpha=0.9,
            label="Null distribution ({:,} permutations)".format(len(perm_r)))
    ax.axvline(r_obs, color="#ff7043", linewidth=2,
               label="Observed r = {:.4f}  (p = {:.4f})".format(r_obs, p_val))
    ax.axvline(min_r, color="#ffeb3b", linewidth=1.5, linestyle="--",
               label="Min. detectable r = {:.4f}  (alpha=0.05, power=0.8)".format(min_r))
    ax.set_xlabel("Mantel r (Pearson)", color=TEXT_COL, fontsize=10)
    ax.set_ylabel("Count", color=TEXT_COL, fontsize=10)
    ax.set_title(title, color=TEXT_COL, fontsize=12, fontweight="bold")
    ax.tick_params(colors=TEXT_COL)
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(SUBTLE)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Mantel distribution plot saved: %s", out_path)


# ===========================================================================
# 2. Robinson-Foulds distance
# ===========================================================================

def compute_robinson_foulds(vis_df, tree_file):
    """
    Compute the symmetric Robinson-Foulds distance between the UPGMA visual
    dendrogram and the molecular tree.

    Both trees share a single TaxonNamespace object, which is required by
    dendropy for bipartition comparison. The visual tree uses underscore-joined
    clean labels (e.g. "A_lineatus") to avoid spaces in Newick format; the
    molecular tree is matched to those same labels via epithet lookup.

    Returns (rf_distance, rf_normalised) or (None, None) on failure.
    """
    try:
        import dendropy
    except ImportError:
        logger.warning("dendropy not installed; skipping Robinson-Foulds.")
        return None, None

    labels    = list(vis_df.index)
    mat       = vis_df.values.copy().astype(float)
    np.fill_diagonal(mat, 0)
    condensed = squareform(mat, checks=False)
    Z         = linkage(condensed, method="average")

    # Build Newick from UPGMA linkage using safe underscore-joined labels
    safe_labels = [clean_label(l).replace(" ", "_").replace(".", "") for l in labels]

    def linkage_to_newick(Z, lbl):
        n     = len(lbl)
        nodes = dict(enumerate(lbl))
        for k, (i, j, dist, _) in enumerate(Z):
            i, j         = int(i), int(j)
            nodes[n + k] = "({i}:{d:.6f},{j}:{d:.6f})".format(
                i=nodes[i], j=nodes[j], d=dist / 2)
        return nodes[n + len(Z) - 1] + ";"

    vis_newick = linkage_to_newick(Z, safe_labels)

    # Shared namespace is essential: without it dendropy raises a namespace
    # mismatch error when comparing bipartitions across the two trees.
    shared_ns = dendropy.TaxonNamespace()

    try:
        vis_tree = dendropy.Tree.get(
            data=vis_newick, schema="newick",
            taxon_namespace=shared_ns
        )
        mol_tree = dendropy.Tree.get(
            path=str(tree_file), schema="newick",
            taxon_namespace=shared_ns
        )
    except Exception as e:
        logger.warning("RF tree loading failed: %s", e)
        return None, None

    # Map visual safe labels to molecular tip labels via epithet matching
    mol_tip_clean = {}
    for leaf in mol_tree.leaf_node_iter():
        raw   = leaf.taxon.label.strip() if leaf.taxon else ""
        clean = raw.replace("_", " ").lower()
        mol_tip_clean[clean] = raw

    retain_raw = set()
    for safe, orig in zip(safe_labels, labels):
        epithet = safe.split("_")[-1].lower()
        for clean, raw in mol_tip_clean.items():
            if clean.split()[-1] == epithet:
                retain_raw.add(raw)
                break

    if not retain_raw:
        logger.warning("RF: no molecular tips matched; skipping.")
        return None, None

    # Prune molecular tree to matched taxa only
    mol_tree.retain_taxa_with_labels(retain_raw)
    mol_tree.purge_taxon_namespace()

    # Prune visual tree to the same safe labels that matched
    matched_safe = set()
    for safe, orig in zip(safe_labels, labels):
        epithet = safe.split("_")[-1].lower()
        for clean in mol_tip_clean:
            if clean.split()[-1] == epithet:
                matched_safe.add(safe)
                break
    vis_tree.retain_taxa_with_labels(matched_safe)
    vis_tree.purge_taxon_namespace()

    try:
        rf = dendropy.calculate.treecompare.symmetric_difference(
            vis_tree, mol_tree, is_bipartitions_updated=False
        )
        n_taxa  = len(labels)
        max_rf  = 2 * (n_taxa - 3)
        rf_norm = float(rf) / max_rf if max_rf > 0 else None
        logger.info("Robinson-Foulds: %d (normalised: %.4f)", rf,
                    rf_norm if rf_norm is not None else 0)
        return int(rf), rf_norm
    except Exception as e:
        logger.warning("RF computation failed: %s", e)
        return None, None


# ===========================================================================
# 3. Visual dendrogram
# ===========================================================================

def plot_visual_dendrogram(vis_df, out_path):
    """
    UPGMA dendrogram of the visual distance matrix. Leaf labels are
    abbreviated (e.g. "A. lineatus") and coloured by genus.

    Genus colours are applied via apply_leaf_colours_from_ivl, which uses
    the dendrogram ivl array as the authoritative leaf sequence and rebuilds
    tick labels from scratch. This is reliable in the Agg backend where
    post-layout colour assignments are otherwise discarded.
    """
    labels    = list(vis_df.index)
    mat       = vis_df.values.copy().astype(float)
    np.fill_diagonal(mat, 0)
    condensed = squareform(mat, checks=False)
    Z         = linkage(condensed, method="average")
    display   = [clean_label(l) for l in labels]

    # colour_map keyed on display (abbreviated) labels
    colour_map = {clean_label(l): genus_colour(l) for l in labels}

    fig_h = max(10, len(labels) * 0.30)
    fig, ax = plt.subplots(figsize=(14, fig_h), facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    ddata = dendrogram(
        Z, labels=display, orientation="right", ax=ax,
        link_color_func=lambda k: BRANCH, leaf_font_size=8,
    )

    ax.set_xlabel("Visual Distance (PCA-Euclidean)", color=TEXT_COL, fontsize=9)
    ax.set_title("Visual Similarity Dendrogram (UPGMA)",
                 color=TEXT_COL, fontsize=12, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_edgecolor(SUBTLE)

    handles = [mpatches.Patch(color=c, label=g)
               for g, c in GENUS_COLOURS.items()]
    ax.legend(handles=handles, fontsize=8, facecolor=PANEL_BG,
              labelcolor=TEXT_COL, loc="lower right")

    fig.tight_layout()

    # Apply genus colours using ivl as the authoritative label order.
    # ivl contains the leaf labels bottom-to-top as rendered; set_yticklabels
    # replaces the tick labels entirely and canvas.draw() forces persistence.
    apply_leaf_colours_from_ivl(ax, ddata["ivl"], colour_map, fontsize=8, axis="y")

    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Visual dendrogram saved: %s", out_path)


# ===========================================================================
# 4. Tanglegram
# ===========================================================================

def plot_tanglegram(vis_df, pat_df, out_path):
    """
    Side-by-side tanglegram with abbreviated, genus-coloured leaf labels.
    Genus colours are applied on both sides using ivl-based label colouring
    after tight_layout, which is the reliable approach in the Agg backend.
    """
    def upgma(df):
        m = df.values.copy().astype(float)
        np.fill_diagonal(m, 0)
        Z     = linkage(squareform(m, checks=False), method="average")
        order = leaves_list(Z)
        lbls  = [list(df.index)[i] for i in order]
        return lbls, Z

    vis_order, Z_vis = upgma(vis_df)
    pat_order, Z_pat = upgma(pat_df)
    n = len(vis_order)

    vis_display = [clean_label(l) for l in vis_order]
    pat_display = [clean_label(l) for l in pat_order]

    vis_colour_map = {clean_label(l): genus_colour(l) for l in vis_order}
    pat_colour_map = {clean_label(l): genus_colour(l) for l in pat_order}

    fig_h = max(12, n * 0.32)
    fig, (ax_vis, ax_mid, ax_pat) = plt.subplots(
        1, 3, figsize=(26, fig_h), facecolor=BG,
        gridspec_kw={"width_ratios": [5, 2, 5]},
    )
    for ax in (ax_vis, ax_mid, ax_pat):
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Left: visual similarity dendrogram
    ddata_vis = dendrogram(
        Z_vis, labels=vis_display, orientation="right", ax=ax_vis,
        link_color_func=lambda k: BRANCH, leaf_font_size=7,
    )
    ax_vis.set_title("Visual Similarity", color=TEXT_COL, fontsize=11,
                     fontweight="bold")
    ax_vis.set_xlabel("Visual Distance", color=TEXT_COL, fontsize=8)
    ax_vis.tick_params(colors=TEXT_COL, length=2)
    ax_vis.yaxis.tick_right()

    # Right: molecular phylogeny dendrogram
    ddata_pat = dendrogram(
        Z_pat, labels=pat_display, orientation="left", ax=ax_pat,
        link_color_func=lambda k: BRANCH, leaf_font_size=7,
    )
    ax_pat.set_title("Molecular Phylogeny", color=TEXT_COL, fontsize=11,
                     fontweight="bold")
    ax_pat.set_xlabel("Patristic Distance (Myr)", color=TEXT_COL, fontsize=8)
    ax_pat.tick_params(colors=TEXT_COL, length=2)
    ax_pat.yaxis.tick_left()

    # Genus legend in the middle panel before layout
    handles = [mpatches.Patch(color=c, label=g)
               for g, c in GENUS_COLOURS.items()]
    ax_mid.axis("off")
    ax_mid.legend(handles=handles, fontsize=7, facecolor=PANEL_BG,
                  labelcolor=TEXT_COL, loc="upper center",
                  bbox_to_anchor=(0.5, 1.04))

    fig.suptitle("Tanglegram: Visual Similarity vs Molecular Phylogeny",
                 color=TEXT_COL, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    # Apply genus colours on both sides using ivl-based approach
    apply_leaf_colours_from_ivl(
        ax_vis, ddata_vis["ivl"], vis_colour_map, fontsize=7, axis="y")
    apply_leaf_colours_from_ivl(
        ax_pat, ddata_pat["ivl"], pat_colour_map, fontsize=7, axis="y")

    # Connecting lines: drawn after layout so positions are finalised
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(-0.5, n - 0.5)

    vis_pos = {clean_label(sp): i for i, sp in enumerate(vis_order)}
    pat_pos = {clean_label(sp): i for i, sp in enumerate(pat_order)}

    for sp in vis_order:
        key = clean_label(sp)
        if key in pat_pos:
            y_l  = vis_pos[key]
            y_r  = pat_pos[key]
            same = (y_l == y_r)
            ax_mid.plot([0, 1], [y_l, y_r],
                        color=genus_colour(sp),
                        alpha=0.75 if same else 0.35,
                        linewidth=1.3 if same else 0.7)

    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Tanglegram saved: %s", out_path)


# ===========================================================================
# 5. Per-feature Mantel correlations
# ===========================================================================

def feature_mantel_correlations(features_csv, pat_df, out_csv, out_png):
    """
    For each of the 99 features, compute Pearson r between the pairwise
    absolute-difference vector and the patristic distance vector.
    Returns a DataFrame sorted by r descending.
    """
    df = pd.read_csv(features_csv)
    meta_cols    = ["genus", "species"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    n_feat       = len(feature_cols)

    pat_species = list(pat_df.index)
    df["_key"]  = (df["genus"] + " " + df["species"]).tolist()
    df_filt     = (df[df["_key"].isin(pat_species)]
                   .set_index("_key").loc[pat_species])

    idx   = np.triu_indices(len(pat_species), k=1)
    p_vec = pat_df.values[idx]

    groups = (FEATURE_GROUP_MAP[:n_feat] +
              ["other"] * max(0, n_feat - len(FEATURE_GROUP_MAP)))

    records = []
    for feat_col, group in zip(feature_cols, groups):
        vals  = df_filt[feat_col].values.astype(float)
        diff  = np.abs(vals[:, None] - vals[None, :])
        v_vec = diff[idx]
        if v_vec.std() < 1e-10:
            r, pv = 0.0, 1.0
        else:
            r, pv = pearsonr(v_vec, p_vec)
        records.append({"feature": feat_col, "group": group,
                         "r": float(r), "p_value": float(pv)})

    feat_df = pd.DataFrame(records).sort_values("r", ascending=False)
    feat_df.to_csv(out_csv, index=False)
    logger.info("Feature correlations saved: %s", out_csv)

    group_order = ["hue_hist", "sat_hist", "val_hist", "dom_colours",
                   "dorsal_ventral", "gabor", "lbp", "entropy"]
    group_mean  = feat_df.groupby("group")["r"].mean().reindex(group_order)
    group_std   = feat_df.groupby("group")["r"].std().reindex(group_order)

    group_colours = {
        "hue_hist"      : "#4fc3f7",
        "sat_hist"      : "#81c784",
        "val_hist"      : "#aed581",
        "dom_colours"   : "#ffb74d",
        "dorsal_ventral": "#f06292",
        "gabor"         : "#ce93d8",
        "lbp"           : "#ff8a65",
        "entropy"       : "#90a4ae",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)

    ax1.set_facecolor(PANEL_BG)
    ax1.bar(range(len(group_order)), group_mean.values,
            yerr=group_std.values,
            color=[group_colours.get(g, "#aaaaaa") for g in group_order],
            edgecolor="none", capsize=4,
            error_kw={"ecolor": TEXT_COL, "alpha": 0.6})
    ax1.axhline(0, color=SUBTLE, linewidth=0.8)
    ax1.set_xticks(range(len(group_order)))
    ax1.set_xticklabels([g.replace("_", "\n") for g in group_order],
                        color=TEXT_COL, fontsize=9)
    ax1.set_ylabel("Mean Mantel r with patristic distance",
                   color=TEXT_COL, fontsize=9)
    ax1.set_title("Phylogenetic Signal by Feature Group",
                  color=TEXT_COL, fontsize=11, fontweight="bold")
    ax1.tick_params(colors=TEXT_COL)
    for sp in ax1.spines.values():
        sp.set_edgecolor(SUBTLE)

    ax2.set_facecolor(PANEL_BG)
    top20 = feat_df.head(20)
    ax2.barh(range(len(top20)), top20["r"].values,
             color=[group_colours.get(g, "#aaaaaa") for g in top20["group"]],
             edgecolor="none")
    ax2.set_yticks(range(len(top20)))
    ax2.set_yticklabels(top20["feature"].tolist(), fontsize=7, color=TEXT_COL)
    ax2.axvline(0, color=SUBTLE, linewidth=0.8)
    ax2.set_xlabel("Mantel r", color=TEXT_COL, fontsize=9)
    ax2.set_title("Top 20 Features by Phylogenetic Signal",
                  color=TEXT_COL, fontsize=11, fontweight="bold")
    ax2.tick_params(colors=TEXT_COL)
    ax2.invert_yaxis()
    for sp in ax2.spines.values():
        sp.set_edgecolor(SUBTLE)

    handles = [mpatches.Patch(color=c, label=g)
               for g, c in group_colours.items()]
    fig.legend(handles=handles, fontsize=8, facecolor=PANEL_BG,
               labelcolor=TEXT_COL, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Feature correlation chart saved: %s", out_png)

    return feat_df


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phylogenetic comparison of visual and molecular distances."
    )
    parser.add_argument(
        "--n-permutations", type=int, default=9999,
        help=("Number of Mantel permutations (default: 9999). "
              "Use 99999 for a stable p-value on borderline results.")
    )
    args = parser.parse_args()

    for p in (VIS_ALIGNED_CSV, VISFS_ALIGNED_CSV, PAT_ALIGNED_CSV):
        if not p.exists():
            logger.error("Required file not found: %s", p)
            logger.error("Run build_distance_matrix.py first.")
            sys.exit(1)

    # Remove stale output from previous script version
    stale = PHYL_DIR / "mantel_permutation_distribution.png"
    if stale.exists():
        stale.unlink()
        logger.info("Removed stale output: %s", stale)

    PHYL_DIR.mkdir(parents=True, exist_ok=True)

    vis_df   = pd.read_csv(VIS_ALIGNED_CSV,   index_col=0)
    visfs_df = pd.read_csv(VISFS_ALIGNED_CSV, index_col=0)
    pat_df   = pd.read_csv(PAT_ALIGNED_CSV,   index_col=0)

    common   = sorted(set(vis_df.index) & set(pat_df.index) & set(visfs_df.index))
    vis_df   = vis_df.loc[common, common]
    visfs_df = visfs_df.loc[common, common]
    pat_df   = pat_df.loc[common, common]
    n        = len(common)

    logger.info("=" * 70)
    logger.info("PHYLOGENETIC COMPARISON")
    logger.info("=" * 70)
    logger.info("Aligned species: %d  |  Permutations: %d",
                n, args.n_permutations)

    vis_mat   = vis_df.values.astype(float);   np.fill_diagonal(vis_mat,   0)
    visfs_mat = visfs_df.values.astype(float); np.fill_diagonal(visfs_mat, 0)
    pat_mat   = pat_df.values.astype(float);   np.fill_diagonal(pat_mat,   0)

    # ------------------------------------------------------------------
    # 1. Mantel tests
    # ------------------------------------------------------------------
    logger.info("Running Mantel test -- PCA matrix (%d permutations)...",
                args.n_permutations)
    res_pca = mantel_test(vis_mat, pat_mat, n_permutations=args.n_permutations)
    logger.info("  Pearson  r=%.4f  p=%.4f", res_pca["r_obs"], res_pca["p_pearson"])
    logger.info("  Spearman r=%.4f  p=%.4f", res_pca["rho_obs"], res_pca["p_spearman"])
    logger.info("  Min detectable r: %.4f  (observed is %s threshold)",
                res_pca["min_detectable_r"],
                "ABOVE" if res_pca["r_obs"] > res_pca["min_detectable_r"] else "BELOW")

    logger.info("Running Mantel test -- feature-selected matrix (%d permutations)...",
                args.n_permutations)
    res_fs = mantel_test(visfs_mat, pat_mat, n_permutations=args.n_permutations)
    logger.info("  Pearson  r=%.4f  p=%.4f", res_fs["r_obs"], res_fs["p_pearson"])
    logger.info("  Spearman r=%.4f  p=%.4f", res_fs["rho_obs"], res_fs["p_spearman"])

    mantel_out = {
        "pca_matrix": {
            "mantel_r"         : res_pca["r_obs"],
            "mantel_rho"       : res_pca["rho_obs"],
            "p_pearson"        : res_pca["p_pearson"],
            "p_spearman"       : res_pca["p_spearman"],
            "n_permutations"   : args.n_permutations,
            "n_species"        : n,
            "n_pairs"          : res_pca["n_pairs"],
            "min_detectable_r" : res_pca["min_detectable_r"],
            "significant_0.05" : res_pca["p_pearson"] < 0.05,
        },
        "feature_selected_matrix": {
            "mantel_r"         : res_fs["r_obs"],
            "mantel_rho"       : res_fs["rho_obs"],
            "p_pearson"        : res_fs["p_pearson"],
            "p_spearman"       : res_fs["p_spearman"],
            "n_permutations"   : args.n_permutations,
            "n_species"        : n,
            "n_pairs"          : res_fs["n_pairs"],
            "min_detectable_r" : res_fs["min_detectable_r"],
            "significant_0.05" : res_fs["p_pearson"] < 0.05,
        },
    }
    with open(PHYL_DIR / "mantel_test_results.json", "w") as f:
        json.dump(mantel_out, f, indent=2)

    plot_mantel_distribution(
        res_pca, "Mantel Test: Visual (PCA) vs Patristic Distance",
        PHYL_DIR / "mantel_permutation_distribution_pca.png"
    )
    plot_mantel_distribution(
        res_fs, "Mantel Test: Visual (Feature-Selected) vs Patristic Distance",
        PHYL_DIR / "mantel_permutation_distribution_featureselected.png"
    )

    # ------------------------------------------------------------------
    # 2. Robinson-Foulds
    # ------------------------------------------------------------------
    logger.info("Computing Robinson-Foulds distance...")
    rf, rf_norm = compute_robinson_foulds(vis_df, TREE_FILE)

    # ------------------------------------------------------------------
    # 3. Visual dendrogram
    # ------------------------------------------------------------------
    logger.info("Generating visual dendrogram...")
    plot_visual_dendrogram(vis_df, PHYL_DIR / "visual_dendrogram.png")

    # ------------------------------------------------------------------
    # 4. Tanglegram
    # ------------------------------------------------------------------
    logger.info("Generating tanglegram...")
    plot_tanglegram(vis_df, pat_df, PHYL_DIR / "tanglegram.png")

    # ------------------------------------------------------------------
    # 5. Per-feature Mantel correlations
    # ------------------------------------------------------------------
    logger.info("Computing per-feature Mantel correlations...")
    feat_df = feature_mantel_correlations(
        FEATURES_CSV, pat_df,
        PHYL_DIR / "feature_mantel_correlations.csv",
        PHYL_DIR / "feature_mantel_correlations.png",
    )

    # ------------------------------------------------------------------
    # 6. Summary JSON
    # ------------------------------------------------------------------
    group_means = feat_df.groupby("group")["r"].mean().to_dict()
    best_group  = max(group_means, key=group_means.get)
    top5        = feat_df.head(5)[["feature", "group", "r",
                                    "p_value"]].to_dict("records")
    summary = {
        "n_species"                       : n,
        "n_permutations"                  : args.n_permutations,
        "mantel_r_pca"                    : res_pca["r_obs"],
        "mantel_p_pca"                    : res_pca["p_pearson"],
        "mantel_rho_pca"                  : res_pca["rho_obs"],
        "mantel_r_featureselected"        : res_fs["r_obs"],
        "mantel_p_featureselected"        : res_fs["p_pearson"],
        "mantel_rho_featureselected"      : res_fs["rho_obs"],
        "pca_significant_0.05"            : res_pca["p_pearson"] < 0.05,
        "featureselected_significant_0.05": res_fs["p_pearson"] < 0.05,
        "min_detectable_r"                : res_pca["min_detectable_r"],
        "robinson_foulds_distance"        : rf,
        "robinson_foulds_normalised"      : rf_norm,
        "highest_signal_group"            : best_group,
        "group_mean_r"                    : group_means,
        "n_features_significant_p05"      : int((feat_df["p_value"] < 0.05).sum()),
        "n_features_negative_r"           : int((feat_df["r"] < 0).sum()),
        "top5_features"                   : top5,
    }
    with open(PHYL_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info("Species: %d  |  Pairs: %d  |  Permutations: %d",
                n, res_pca["n_pairs"], args.n_permutations)
    logger.info("")
    logger.info("Mantel test (PCA matrix):")
    logger.info("  Pearson  r=%.4f  p=%.4f  significant=%s",
                res_pca["r_obs"], res_pca["p_pearson"],
                res_pca["p_pearson"] < 0.05)
    logger.info("  Spearman r=%.4f  p=%.4f",
                res_pca["rho_obs"], res_pca["p_spearman"])
    logger.info("  Min detectable r: %.4f  (observed is %s threshold)",
                res_pca["min_detectable_r"],
                "ABOVE" if res_pca["r_obs"] > res_pca["min_detectable_r"]
                else "BELOW")
    logger.info("")
    logger.info("Mantel test (feature-selected matrix):")
    logger.info("  Pearson  r=%.4f  p=%.4f  significant=%s",
                res_fs["r_obs"], res_fs["p_pearson"],
                res_fs["p_pearson"] < 0.05)
    logger.info("  Spearman r=%.4f  p=%.4f",
                res_fs["rho_obs"], res_fs["p_spearman"])
    logger.info("")
    if rf is not None:
        logger.info("Robinson-Foulds: %d (normalised: %.4f)", rf, rf_norm)
    else:
        logger.info("Robinson-Foulds: not computed (see warnings above)")
    logger.info("")
    logger.info("Feature group mean Mantel r (ranked):")
    for grp in sorted(group_means, key=group_means.get, reverse=True):
        logger.info("  %-20s  r=%+.4f", grp, group_means[grp])
    logger.info("")
    logger.info("Features significant at p<0.05: %d / 99",
                int((feat_df["p_value"] < 0.05).sum()))
    logger.info("Features with negative r:        %d / 99",
                int((feat_df["r"] < 0).sum()))
    logger.info("")
    logger.info("Outputs: %s", PHYL_DIR)


if __name__ == "__main__":
    main()