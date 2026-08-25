# Phase 4 primary test: Kmult phylogenetic signal via geomorph::physignal.z(),
# compared across dimensions via geomorph::compare.physignal.z().
#
# ============================================================================
# IMPORTANT: this script has NOT been executed or verified against a real R
# installation. Every other piece of code in this project (Python) was run
# and its output checked against real data/ground truth before being trusted
# - this file is the one exception, because no R runtime is available in the
# environment that wrote it. Run it cell-by-cell (or step-by-step) the first
# time, actually look at each intermediate print()/str() output, and treat
# the final p-values as provisional until you've done that - the same
# discipline every other phase in this project went through, not skipped
# here just because the language changed.
# ============================================================================
#
# Inputs (written by `python -m phylo_comparison.pipeline`, see
# src/phylo_comparison/): outputs/phase4/{color,stripe,spot}_kmult_features.csv
# (species x feature, already transformed/standardized in Python - see
# src/phylo_comparison/feature_prep.py's docstring for the admissibility
# reasoning) and outputs/phase4/pruned_tree.nwk (50-ish tips, labels already
# renamed to match the "species" column in every feature CSV exactly).
#
# Preregistered settings (see README.md's Planned Approach step 4 for the
# full reasoning behind each, checked across five external audit rounds
# against the real geomorph manual before being pinned):
#   - geomorph::physignal.z(), not physignal() - physignal() no longer
#     reports an effect size as of geomorph 4.0.2 and defers to physignal.z.
#   - lambda = "front" (not "burn" or "all" - see README for why).
#   - PAC.no = the dimension's own full feature count (no PCA reduction -
#     already far below n=49 species, see src/phylo_comparison's package
#     docstring for why the plan's shape-data-many-landmarks assumption
#     doesn't apply to this feature set).
#   - seed = NULL (physignal.z's deterministic default - same P-values on
#     repeated runs).
#   - compare.physignal.z() across all three dimensions' results, for
#     "which pattern dimension carries the most signal" - three p-values
#     alone can't answer that since raw K isn't comparable across matrices.
#   - Benjamini-Hochberg (FDR) correction across the three dimensions'
#     p-values, applied to the primary Kmult results (the plan requires a
#     multiple-comparisons correction; BH is used here as a standard,
#     less conservative default - swap for p.adjust(method = "bonferroni")
#     if a more conservative correction is preferred).
#
# Secondary test (Steps 6-8): phylogenetically-permuted Mantel check per
# dimension, using Phase 3's already-built pattern-distance matrices
# (outputs/{color,stripe,spot}_distance_matrix.csv) against its patristic
# distance matrix (outputs/patristic_distance_matrix.csv). Per Harmon &
# Glor (2010) - the paper that motivated this project's whole
# Kmult-over-Mantel decision - naive label-shuffling permutation inflates
# Mantel's type-I error when both matrices carry phylogenetic
# autocorrelation; their remedy is to build the null distribution by
# simulating trait evolution ON THE REAL TREE (Brownian motion) rather
# than permuting species labels. Implemented here using ape::rTraitCont()
# (its real signature confirmed against the CRAN manual before writing
# this - not guessed, see CLAUDE.md's "no guessing on unfamiliar
# library/API surfaces" rule, added after this script's first draft
# guessed a geomorph field name and crashed on real data) plus base R's
# cor()/lower.tri()/dist() for the Mantel statistic itself, deliberately
# avoiding another unfamiliar package's Mantel implementation (e.g.
# vegan::mantel()) where the risk of a repeat guessing mistake is real.
#
# THIS SECTION (Steps 6-8) IS NEWER AND LESS-TESTED THAN STEPS 0-5, WHICH
# HAVE ALREADY BEEN RUN SUCCESSFULLY ON REAL DATA. It has not yet been
# executed at all - same caveat as this whole file originally carried,
# now narrowed to just this addition.

library(geomorph)
library(ape)

# ---------------------------------------------------------------------------
# Step 0: smoke-test this R/geomorph installation against the package's own
# documented example (data(plethspecies)) before trusting it on real data -
# not the numerical-equivalence gate the README specifies for a *Python
# port* of physignal.z (this script isn't a port, it calls the real
# function), but the same spirit: confirm the installed geomorph actually
# behaves the way its own manual says it does, on data whose expected shape
# is known, before trusting unfamiliar output on the real study data.
# ---------------------------------------------------------------------------
cat("=== Step 0: smoke test against geomorph::plethspecies ===\n")
data(plethspecies)
Y.gpa <- gpagen(plethspecies$land, print.progress = FALSE)
smoke_test <- physignal.z(
  A = Y.gpa$coords, phy = plethspecies$phy,
  lambda = "front", PAC.no = 7, seed = NULL
)
print(smoke_test)
cat("Compare the printed K/Z/P above against the geomorph manual's own\n")
cat("physignal.z example output before proceeding - if these don't match,\n")
cat("stop and fix the R/geomorph installation, don't continue to real data.\n\n")

# ---------------------------------------------------------------------------
# Step 1: load the pruned tree and every dimension's prepared feature matrix.
# ---------------------------------------------------------------------------
cat("=== Step 1: load tree and feature matrices ===\n")
phy <- read.tree("outputs/phase4/pruned_tree.nwk")
cat(sprintf("Tree: %d tips\n", length(phy$tip.label)))

dimensions <- c("color", "stripe", "spot")
data_by_dimension <- list()
for (dim in dimensions) {
  path <- sprintf("outputs/phase4/%s_kmult_features.csv", dim)
  df <- read.csv(path, row.names = "species")
  data_by_dimension[[dim]] <- as.matrix(df)
  cat(sprintf("%s: %d species x %d features\n", dim, nrow(df), ncol(df)))
}

# ---------------------------------------------------------------------------
# Step 2: integrity checks before running anything - fail loudly on a
# mismatch rather than let physignal.z silently drop or misalign rows.
# ---------------------------------------------------------------------------
cat("\n=== Step 2: integrity checks ===\n")
for (dim in dimensions) {
  data_species <- sort(rownames(data_by_dimension[[dim]]))
  tree_species <- sort(phy$tip.label)
  if (!identical(data_species, tree_species)) {
    stop(sprintf(
      "%s: feature matrix species don't match tree tips exactly.\nIn data not tree: %s\nIn tree not data: %s",
      dim,
      paste(setdiff(data_species, tree_species), collapse = ", "),
      paste(setdiff(tree_species, data_species), collapse = ", ")
    ))
  }
}
cat("OK: every dimension's feature-matrix species exactly match the tree's tips.\n\n")

# ---------------------------------------------------------------------------
# Step 3: primary test - physignal.z per dimension.
# ---------------------------------------------------------------------------
cat("=== Step 3: physignal.z per dimension ===\n")
physignal_results <- list()
for (dim in dimensions) {
  mat <- data_by_dimension[[dim]]
  pac_no <- ncol(mat)  # no PCA reduction - see the header comment for why
  cat(sprintf("--- %s (PAC.no = %d) ---\n", dim, pac_no))
  result <- physignal.z(A = mat, phy = phy, lambda = "front", PAC.no = pac_no, seed = NULL)
  print(result)
  if (is.nan(result$Z)) {
    cat(sprintf(
      "WARNING: %s produced Z = NaN - per the geomorph manual, this happens when the\n",
      dim
    ))
    cat("phylogenetic scaling parameter (lambda) optimized to 0; pairwise comparisons\n")
    cat("involving this dimension 'might not make sense' (manual's own wording).\n")
  }
  physignal_results[[dim]] <- result
  cat("\n")
}

# ---------------------------------------------------------------------------
# Step 4: cross-dimension comparison - which pattern dimension carries the
# most signal, not answerable from three separate p-values alone.
# ---------------------------------------------------------------------------
cat("=== Step 4: compare.physignal.z across dimensions ===\n")
comparison <- compare.physignal.z(
  physignal_results[["color"]], physignal_results[["stripe"]], physignal_results[["spot"]]
)
print(comparison)
cat("\n")

# ---------------------------------------------------------------------------
# Step 5: multiple-comparisons correction across the three dimensions'
# physignal.z p-values (Benjamini-Hochberg / FDR - see header comment).
# ---------------------------------------------------------------------------
cat("=== Step 5: multiple-comparisons correction ===\n")
raw_p <- sapply(physignal_results, function(r) r$pvalue)
corrected_p <- p.adjust(raw_p, method = "BH")
results_table <- data.frame(
  dimension = dimensions,
  K = sapply(physignal_results, function(r) r$K),
  Z = sapply(physignal_results, function(r) r$Z),
  raw_p = raw_p,
  bh_corrected_p = corrected_p
)
print(results_table)
cat("\nA null result (non-significant, corrected p) is inconclusive, not evidence\n")
cat("of no association - Harmon & Glor (2010) found real power limitations even\n")
cat("with the correct phylogenetic-signal remedy applied. See README.md.\n\n")

write.csv(results_table, "outputs/phase4/kmult_results.csv", row.names = FALSE)
cat("Wrote outputs/phase4/kmult_results.csv\n\n")

# ---------------------------------------------------------------------------
# Step 6: secondary test - phylogenetically-permuted Mantel check per
# dimension. See the header comment for the full method and why
# ape::rTraitCont() (verified) plus base R (not another unfamiliar
# package's Mantel implementation) was used.
# ---------------------------------------------------------------------------
cat("=== Step 6: secondary test - phylogenetically-permuted Mantel ===\n")

mantel_r <- function(dist_a, dist_b) {
  # Pearson correlation between corresponding lower-triangle entries of two
  # distance matrices - the standard modern Mantel statistic definition
  # (as used by e.g. vegan::mantel()), not Mantel's original (1967)
  # unnormalized cross-product sum.
  idx <- lower.tri(dist_a)
  cor(dist_a[idx], dist_b[idx])
}

n_permutations <- 1000
set.seed(1)  # fixed seed - the simulated null distribution below is stochastic,
             # unlike physignal.z's own RRPP scheme (seed = NULL is already
             # deterministic there); without a fixed seed here, re-running
             # this section would jitter the Mantel p-values slightly each time.

patristic <- as.matrix(read.csv("outputs/patristic_distance_matrix.csv", row.names = 1))
species_order <- rownames(patristic)

if (!identical(sort(species_order), sort(phy$tip.label))) {
  stop(sprintf(
    "Phase 3's patristic matrix species don't match Phase 4's tree tips exactly.\nIn patristic not tree: %s\nIn tree not patristic: %s",
    paste(setdiff(species_order, phy$tip.label), collapse = ", "),
    paste(setdiff(phy$tip.label, species_order), collapse = ", ")
  ))
}

mantel_results <- list()
for (dim in dimensions) {
  pattern_dist <- as.matrix(read.csv(sprintf("outputs/%s_distance_matrix.csv", dim), row.names = 1))
  if (!identical(sort(rownames(pattern_dist)), sort(species_order))) {
    stop(sprintf(
      "%s_distance_matrix.csv species don't match the patristic matrix exactly.\nIn pattern not patristic: %s\nIn patristic not pattern: %s",
      dim,
      paste(setdiff(rownames(pattern_dist), species_order), collapse = ", "),
      paste(setdiff(species_order, rownames(pattern_dist)), collapse = ", ")
    ))
  }
  pattern_dist <- pattern_dist[species_order, species_order]

  observed_r <- mantel_r(pattern_dist, patristic)

  n_features <- ncol(data_by_dimension[[dim]])  # same dimensionality Step 3 used for physignal.z
  null_r <- numeric(n_permutations)
  for (i in seq_len(n_permutations)) {
    sim_traits <- sapply(seq_len(n_features), function(j) rTraitCont(phy, model = "BM"))
    rownames(sim_traits) <- phy$tip.label  # explicit, not assumed - see header comment
    sim_dist <- as.matrix(dist(sim_traits))
    sim_dist <- sim_dist[species_order, species_order]
    null_r[i] <- mantel_r(sim_dist, patristic)
  }

  p_value <- mean(abs(null_r) >= abs(observed_r))
  mantel_results[[dim]] <- list(observed_r = observed_r, p_value = p_value)
  cat(sprintf(
    "%s: observed Mantel r = %.4f, phylogenetically-permuted p = %.4f (n=%d sims)\n",
    dim, observed_r, p_value, n_permutations
  ))
}
cat("\n")

# ---------------------------------------------------------------------------
# Step 7: multiple-comparisons correction across the three dimensions'
# Mantel p-values (same BH method as Step 5, per the README's requirement
# that this apply to both the primary and secondary tests).
# ---------------------------------------------------------------------------
cat("=== Step 7: Mantel multiple-comparisons correction ===\n")
mantel_raw_p <- sapply(mantel_results, function(r) r$p_value)
mantel_corrected_p <- p.adjust(mantel_raw_p, method = "BH")
mantel_table <- data.frame(
  dimension = dimensions,
  observed_r = sapply(mantel_results, function(r) r$observed_r),
  raw_p = mantel_raw_p,
  bh_corrected_p = mantel_corrected_p
)
print(mantel_table)
cat("\nSame caveat as the primary test: a null result here is inconclusive, not\n")
cat("evidence of no association.\n\n")

# ---------------------------------------------------------------------------
# Step 8: write the secondary results.
# ---------------------------------------------------------------------------
write.csv(mantel_table, "outputs/phase4/mantel_results.csv", row.names = FALSE)
cat("Wrote outputs/phase4/mantel_results.csv\n")
