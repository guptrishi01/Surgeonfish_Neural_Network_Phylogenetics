"""
build_distance_matrix.py

Computes two distance matrices and saves them as CSVs with visualisations:

  1. Visual distance matrix  -- Euclidean distance in PCA-reduced feature
	 space (95% variance threshold), plus a second feature-selected matrix
	 using only the four feature groups with positive mean phylogenetic signal
	 (dorsal_ventral, hue_hist, entropy, sat_hist: 37 features total).

  2. Patristic distance matrix -- cophenetic branch-length distances between
	 all species pairs extracted from the Fish Tree of Life Newick tree.

A matching audit CSV is written documenting every visual species and whether
it was found in the tree, under what name, and via which matching method.
Dropped species are also saved to a plain-text file for review.

Outputs
-------
  outputs/distance_matrices/visual_distance_matrix.csv
  outputs/distance_matrices/visual_distance_matrix_featureselected.csv
  outputs/distance_matrices/patristic_distance_matrix.csv
  outputs/distance_matrices/visual_distance_matrix_aligned.csv
  outputs/distance_matrices/visual_distance_matrix_featureselected_aligned.csv
  outputs/distance_matrices/patristic_distance_matrix_aligned.csv
  outputs/distance_matrices/visual_distance_heatmap.png
  outputs/distance_matrices/patristic_distance_heatmap.png
  outputs/distance_matrices/pca_variance_explained.png
  outputs/distance_matrices/tree_tip_matching_audit.csv
  outputs/distance_matrices/dropped_species.txt
  outputs/distance_matrices/distance_matrix_summary.json

Usage
-----
  python scripts/python/build_distance_matrix.py
  python scripts/python/build_distance_matrix.py --inspect-tree
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUT_DIR      = PROJECT_ROOT / "outputs"
DIST_DIR     = OUT_DIR / "distance_matrices"

FEATURES_CSV = DATA_DIR / "features" / "features.csv"
TREE_FILE    = DATA_DIR / "phylogeny" / "Acanthuridae_timetree.tre"

# Feature groups with demonstrated positive phylogenetic signal.
# These are used for the feature-selected distance matrix.
SIGNAL_GROUPS = {"dorsal_ventral", "hue_hist", "entropy", "sat_hist"}

# Feature group assignment matching extract_features.py order (99 features).
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

BG       = "#0f1117"
PANEL_BG = "#1a1d27"
TEXT_COL = "#e8e8f0"
SUBTLE   = "#4a4d5e"
ACCENT   = "#4fc3f7"


# ===========================================================================
# Helpers
# ===========================================================================

def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
	"""Return a symmetric (n x n) Euclidean distance matrix for rows of X."""
	n    = X.shape[0]
	dist = np.zeros((n, n))
	for i in range(n):
		for j in range(i + 1, n):
			d = np.linalg.norm(X[i] - X[j])
			dist[i, j] = d
			dist[j, i] = d
	return dist


# ===========================================================================
# 1. Visual distance matrices
# ===========================================================================

def build_visual_distance_matrices(features_csv: Path):
	"""
	Load features.csv and produce two distance matrices:

	  (a) PCA-based: standardise all 99 features, reduce to 95% variance via
		  PCA, then compute pairwise Euclidean distances.

	  (b) Feature-selected: standardise only the features belonging to the
		  four positive-signal groups (dorsal_ventral, hue_hist, entropy,
		  sat_hist), then compute pairwise Euclidean distances without PCA.
		  This avoids the signal-cancellation effect caused by including the
		  LBP and val_hist groups, which carry negative mean phylogenetic r.

	Returns (pca_dist_df, fs_dist_df, pca_full, n_comp, labels).
	"""
	logger.info("Loading feature matrix from %s", features_csv)
	df = pd.read_csv(features_csv)

	meta_cols    = ["genus", "species"]
	feature_cols = [c for c in df.columns if c not in meta_cols]
	n_feat       = len(feature_cols)
	logger.info("Feature matrix: %d species x %d features", len(df), n_feat)

	labels = (df["genus"] + " " + df["species"]).tolist()
	X      = df[feature_cols].values.astype(np.float64)

	# (a) PCA-based distance matrix
	scaler  = StandardScaler()
	X_std   = scaler.fit_transform(X)

	pca_full = PCA().fit(X_std)
	cum_var  = np.cumsum(pca_full.explained_variance_ratio_)
	n_comp   = int(np.searchsorted(cum_var, 0.95)) + 1
	n_comp   = min(n_comp, len(labels) - 1)
	logger.info("PCA: %d components retain %.1f%% variance",
				n_comp, cum_var[n_comp - 1] * 100)

	pca   = PCA(n_components=n_comp)
	X_pca = pca.fit_transform(X_std)

	pca_dist_df = pd.DataFrame(
		pairwise_euclidean(X_pca), index=labels, columns=labels
	)

	# (b) Feature-selected distance matrix
	groups = FEATURE_GROUP_MAP[:n_feat]
	sel_idx = [i for i, g in enumerate(groups) if g in SIGNAL_GROUPS]
	logger.info("Feature-selected matrix: %d features from groups %s",
				len(sel_idx), sorted(SIGNAL_GROUPS))

	X_sel     = X[:, sel_idx]
	scaler_fs = StandardScaler()
	X_sel_std = scaler_fs.fit_transform(X_sel)

	fs_dist_df = pd.DataFrame(
		pairwise_euclidean(X_sel_std), index=labels, columns=labels
	)

	logger.info("PCA distance range:   [%.3f, %.3f]",
				pca_dist_df.values[~np.eye(len(labels), dtype=bool)].min(),
				pca_dist_df.values.max())
	logger.info("Feature-sel distance range: [%.3f, %.3f]",
				fs_dist_df.values[~np.eye(len(labels), dtype=bool)].min(),
				fs_dist_df.values.max())

	return pca_dist_df, fs_dist_df, pca_full, n_comp, labels


# ===========================================================================
# 2. Patristic distance matrix
# ===========================================================================

def build_patristic_distance_matrix(tree_file: Path, visual_labels: list,
									 dist_dir: Path, inspect_only: bool = False):
	"""
	Extract pairwise patristic distances from the Newick tree using dendropy.

	Matching strategy (applied in order, first success wins):
	  1. Exact match after replacing underscores with spaces.
	  2. Match against the last two words of the visual label (handles the
		 double-genus format "Acanthurus Acanthurus bariene" -> "Acanthurus bariene").
	  3. Match against the species epithet only (last word).
	  4. No match -- species recorded as absent from tree.

	A full audit CSV is written documenting the outcome for every species.

	Returns (dist_df, matched_visual_labels) or exits if inspect_only=True.
	"""
	try:
		import dendropy
	except ImportError:
		logger.error("dendropy not installed. Run: pip install dendropy --break-system-packages")
		sys.exit(1)

	logger.info("Loading phylogenetic tree from %s", tree_file)
	tree = dendropy.Tree.get(path=str(tree_file), schema="newick")

	# Build a lookup of cleaned tip label -> raw label
	tree_taxa_clean = {}   # cleaned (spaces) -> raw
	tree_taxa_raw   = {}   # raw -> taxon object
	for leaf in tree.leaf_node_iter():
		raw   = leaf.taxon.label.strip() if leaf.taxon else ""
		clean = raw.replace("_", " ")
		tree_taxa_clean[clean] = raw
		tree_taxa_raw[raw]     = leaf.taxon

	logger.info("Tree contains %d tips", len(tree_taxa_clean))

	if inspect_only:
		print("\nAll tree tip labels (cleaned):")
		for c in sorted(tree_taxa_clean.keys()):
			print(f"  {c!r}")
		sys.exit(0)

	# Match visual labels to tree tips
	audit_rows         = []
	matched_visual     = []
	matched_tree_raw   = []

	for label in visual_labels:
		parts   = label.split()
		epithet = parts[-1] if parts else ""
		last_two = " ".join(parts[-2:]) if len(parts) >= 2 else label

		result = None

		# Strategy 1: exact
		if label in tree_taxa_clean:
			result = ("exact", tree_taxa_clean[label])

		# Strategy 2: last two words (handles "Genus Genus species" format)
		if result is None and last_two in tree_taxa_clean:
			result = ("last_two_words", tree_taxa_clean[last_two])

		# Strategy 3: epithet only
		if result is None:
			epithet_lower = epithet.lower()
			for clean, raw in tree_taxa_clean.items():
				if clean.split()[-1].lower() == epithet_lower:
					result = ("epithet_only", raw)
					break
		# Strategy 4: strip underscores from epithet and match on last word
		if result is None:
			epithet_clean = epithet.replace("_", " ").split()[-1].lower()
			for clean, raw in tree_taxa_clean.items():
				if clean.split()[-1].lower() == epithet_clean:
					result = ("epithet_underscore_stripped", raw)
					break

		if result is not None:
			method, raw = result
			matched_visual.append(label)
			matched_tree_raw.append(raw)
			audit_rows.append({
				"visual_label"     : label,
				"matched_tree_tip" : raw,
				"match_method"     : method,
				"in_tree"          : True,
			})
		else:
			audit_rows.append({
				"visual_label"     : label,
				"matched_tree_tip" : "",
				"match_method"     : "none",
				"in_tree"          : False,
			})

	# Save audit
	audit_df   = pd.DataFrame(audit_rows)
	audit_path = dist_dir / "tree_tip_matching_audit.csv"
	audit_df.to_csv(audit_path, index=False)
	logger.info("Tree tip matching audit saved: %s", audit_path)

	# Save dropped species
	dropped     = audit_df[~audit_df["in_tree"]]["visual_label"].tolist()
	dropped_path = dist_dir / "dropped_species.txt"
	dropped_path.write_text("\n".join(dropped) + "\n", encoding="utf-8")
	logger.info("Dropped species (%d): %s", len(dropped), dropped_path)
	if dropped:
		for sp in dropped:
			logger.warning("  Not in tree: %s", sp)

	logger.info("Matched %d / %d species to tree tips",
				len(matched_visual), len(visual_labels))

	# Compute patristic distances
	pdm      = tree.phylogenetic_distance_matrix()
	taxon_map = {t.label: t for t in tree.taxon_namespace}

	n    = len(matched_visual)
	dist = np.zeros((n, n))

	for i, raw_i in enumerate(matched_tree_raw):
		for j, raw_j in enumerate(matched_tree_raw):
			if i == j:
				continue
			ti = taxon_map.get(raw_i)
			tj = taxon_map.get(raw_j)
			if ti is None or tj is None:
				dist[i, j] = np.nan
				continue
			try:
				dist[i, j] = pdm(ti, tj)
			except Exception:
				dist[i, j] = np.nan

	dist_df = pd.DataFrame(dist, index=matched_visual, columns=matched_visual)
	logger.info("Patristic distance matrix: %dx%d, range [%.3f, %.3f]",
				n, n, np.nanmin(dist[dist > 0]), np.nanmax(dist))

	return dist_df, set(matched_visual)


# ===========================================================================
# 3. Visualisation helpers
# ===========================================================================

def plot_heatmap(dist_df: pd.DataFrame, title: str, out_path: Path,
				 cmap: str = "viridis"):
	"""Save a hierarchically-reordered heatmap of a distance matrix."""
	labels = list(dist_df.index)
	mat    = dist_df.values.copy().astype(float)
	np.fill_diagonal(mat, 0)

	condensed = squareform(mat, checks=False)
	Z         = linkage(condensed, method="average")
	order     = leaves_list(Z)
	mat       = mat[np.ix_(order, order)]
	labels    = [labels[i] for i in order]

	n     = len(labels)
	fig_w = max(14, n * 0.22)
	fig_h = max(12, n * 0.20)

	fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
	ax.set_facecolor(PANEL_BG)

	im   = ax.imshow(mat, cmap=cmap, aspect="auto")
	cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
	cbar.ax.tick_params(colors=TEXT_COL, labelsize=7)
	cbar.set_label("Distance", color=TEXT_COL, fontsize=9)

	ax.set_xticks(range(n))
	ax.set_yticks(range(n))
	ax.set_xticklabels(labels, rotation=90, fontsize=6, color=TEXT_COL)
	ax.set_yticklabels(labels, fontsize=6, color=TEXT_COL)
	for spine in ax.spines.values():
		spine.set_edgecolor(SUBTLE)

	ax.set_title(title, color=TEXT_COL, fontsize=12, fontweight="bold", pad=10)
	fig.tight_layout()
	fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=BG)
	plt.close(fig)
	logger.info("Heatmap saved: %s", out_path)


def plot_pca_variance(pca_full, n_comp: int, out_path: Path):
	"""Save a scree plot showing cumulative variance explained."""
	evr     = pca_full.explained_variance_ratio_
	cum_var = np.cumsum(evr)
	n_show  = min(30, len(evr))

	fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
	ax.set_facecolor(PANEL_BG)

	ax.bar(range(1, n_show + 1), evr[:n_show] * 100,
		   color=ACCENT, alpha=0.7, label="Individual")
	ax.plot(range(1, n_show + 1), cum_var[:n_show] * 100,
			color="#ff7043", marker="o", ms=4, linewidth=1.5,
			label="Cumulative")
	ax.axvline(n_comp, color="white", linestyle="--", linewidth=1,
			   label=f"Selected ({n_comp} comps, {cum_var[n_comp-1]*100:.1f}%)")
	ax.axhline(95, color="#aaaaaa", linestyle=":", linewidth=0.8)

	ax.set_xlabel("Principal Component", color=TEXT_COL, fontsize=10)
	ax.set_ylabel("Variance Explained (%)", color=TEXT_COL, fontsize=10)
	ax.set_title("PCA Scree Plot -- Visual Feature Space",
				 color=TEXT_COL, fontsize=12, fontweight="bold")
	ax.tick_params(colors=TEXT_COL)
	ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL)
	for spine in ax.spines.values():
		spine.set_edgecolor(SUBTLE)

	fig.tight_layout()
	fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=BG)
	plt.close(fig)
	logger.info("PCA scree plot saved: %s", out_path)


# ===========================================================================
# Main
# ===========================================================================

def main():
	parser = argparse.ArgumentParser(
		description="Build visual and patristic distance matrices."
	)
	parser.add_argument(
		"--inspect-tree", action="store_true",
		help="Print all tree tip labels and exit (useful for auditing matches)."
	)
	args = parser.parse_args()

	logger.info("=" * 70)
	logger.info("BUILD DISTANCE MATRICES")
	logger.info("=" * 70)

	DIST_DIR.mkdir(parents=True, exist_ok=True)

	# ------------------------------------------------------------------
	# 1. Visual distance matrices
	# ------------------------------------------------------------------
	pca_dist_df, fs_dist_df, pca_full, n_comp, labels = \
		build_visual_distance_matrices(FEATURES_CSV)

	pca_dist_df.to_csv(DIST_DIR / "visual_distance_matrix.csv")
	fs_dist_df.to_csv(DIST_DIR / "visual_distance_matrix_featureselected.csv")
	logger.info("Visual distance matrices saved.")

	plot_pca_variance(pca_full, n_comp, DIST_DIR / "pca_variance_explained.png")
	plot_heatmap(pca_dist_df,
				 "Visual Distance Matrix (PCA-Euclidean)",
				 DIST_DIR / "visual_distance_heatmap.png", cmap="plasma")
	plot_heatmap(fs_dist_df,
				 "Visual Distance Matrix (Feature-Selected: dorsal_ventral + hue_hist + entropy + sat_hist)",
				 DIST_DIR / "visual_distance_heatmap_featureselected.png", cmap="plasma")

	# ------------------------------------------------------------------
	# 2. Patristic distance matrix
	# ------------------------------------------------------------------
	pat_dist_df, matched_species = build_patristic_distance_matrix(
		TREE_FILE, labels, DIST_DIR, inspect_only=args.inspect_tree
	)

	pat_dist_df.to_csv(DIST_DIR / "patristic_distance_matrix.csv")
	logger.info("Patristic distance matrix saved.")

	plot_heatmap(pat_dist_df,
				 "Patristic Distance Matrix (Fish Tree of Life)",
				 DIST_DIR / "patristic_distance_heatmap.png", cmap="viridis")

	# ------------------------------------------------------------------
	# 3. Align all matrices to the common species set
	# ------------------------------------------------------------------
	common = sorted(set(pca_dist_df.index) & set(pat_dist_df.index))
	logger.info("Species in all aligned matrices: %d", len(common))

	pca_dist_df.loc[common, common].to_csv(
		DIST_DIR / "visual_distance_matrix_aligned.csv")
	fs_dist_df.loc[common, common].to_csv(
		DIST_DIR / "visual_distance_matrix_featureselected_aligned.csv")
	pat_dist_df.loc[common, common].to_csv(
		DIST_DIR / "patristic_distance_matrix_aligned.csv")
	logger.info("Aligned matrices saved.")

	# ------------------------------------------------------------------
	# 4. Summary JSON
	# ------------------------------------------------------------------
	summary = {
		"n_species_visual"          : len(pca_dist_df),
		"n_species_patristic"       : len(pat_dist_df),
		"n_species_aligned"         : len(common),
		"n_species_dropped"         : len(labels) - len(common),
		"pca_n_components"          : int(n_comp),
		"pca_variance_retained"     : float(
			np.cumsum(pca_full.explained_variance_ratio_)[n_comp - 1]),
		"n_featureselected_features": int(sum(
			1 for g in FEATURE_GROUP_MAP if g in SIGNAL_GROUPS)),
		"featureselected_groups"    : sorted(SIGNAL_GROUPS),
		"visual_dist_min"           : float(
			pca_dist_df.values[~np.eye(len(pca_dist_df), dtype=bool)].min()),
		"visual_dist_max"           : float(pca_dist_df.values.max()),
		"patristic_dist_min"        : float(
			pat_dist_df.values[~np.eye(len(pat_dist_df), dtype=bool)][
				pat_dist_df.values[~np.eye(len(pat_dist_df), dtype=bool)] > 0].min()),
		"patristic_dist_max"        : float(pat_dist_df.values.max()),
		"aligned_species"           : common,
	}

	with open(DIST_DIR / "distance_matrix_summary.json", "w") as f:
		json.dump(summary, f, indent=2)

	logger.info("Summary JSON saved.")
	logger.info("")
	logger.info("Done. All outputs in: %s", DIST_DIR)
	logger.info("Next step: python scripts/python/compare_to_phylogeny.py")


if __name__ == "__main__":
	main()