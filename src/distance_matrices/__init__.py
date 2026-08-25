"""Phase 3 of the surgeonfish ML pipeline: per-species aggregation and distance matrices.

Collapses ``pattern_extractor``'s per-image feature rows (``reports/
pattern_features.csv``) into one feature vector per species, per pattern
dimension (coloring, stripe, spot), restricted to the 50 species with real
genetic-data phylogenetic placement (``data/phylogeny/species_coverage.csv``
- see that file's own ``README.md`` for the restriction's justification).
Produces three species x species visual-distance matrices (one per pattern
dimension) plus one patristic-distance matrix from the pruned molecular
tree, for the secondary phylogenetically-permuted Mantel check described in
the project README's Planned Approach step 4. The primary test (Kmult, via
``geomorph::physignal.z`` in R) is Phase 4, not built here - this phase's
output (the per-species feature matrices) is what Phase 4 consumes.

**Aggregation choices, and why:**

- **`is_reference` rows excluded by default.** The curated seed photo
  (``000_reference``) was hand-picked to seed `pattern_extractor`'s
  clustering, not sampled the way GBIF field photos were - including it
  risks a systematic bias distinct from the rest of a species' images.
  The one species this matters most for, *Naso maculatus*, has exactly one
  image total and it *is* the reference photo (confirmed directly against
  ``data/raw_images/Naso/Naso_maculatus.zip`` during Phase 2's real-data
  review) - excluding reference rows drops that species from the analysis
  entirely (50 -> 49), a real, documented consequence of this choice, not
  an oversight. Configurable (`AggregationConfig.include_reference_images`)
  so the alternative can be run as an explicit sensitivity check.
- **Per-image feature averaging, not a single combined statistic.** Every
  numeric feature `pattern_extractor` outputs is a non-circular scalar
  (proportions, counts, an FFT peak-strength ratio, a hue/saturation
  dispersion magnitude) - none of them are a raw hue *angle*, so the
  circular-mean handling the original plan anticipated turned out not to
  be needed once the actual feature set was implemented; a plain
  arithmetic mean is correct for all of them. Boolean features
  (`is_solid`/`stripe_present`/`spot_present`) are aggregated as the
  *proportion* of a species' images scoring True, not a majority-vote
  boolean - this preserves more information (a species at 90% striped
  images looks different from one at 55%, even though both would collapse
  to the same hard "True" under a vote) and matches how repeated binary
  measurements are normally summarized.
- **`mean_spot_area` is normalized before averaging.** `fish_extractor`
  deliberately doesn't resize crops, so a raw pixel-area feature isn't
  comparable across images of different native resolutions - two
  biologically-identical spots photographed at different zoom/resolution
  would report different `mean_spot_area` values for reasons that have
  nothing to do with the fish. `aggregation.py` re-derives each image's
  total masked-pixel count directly from its mask file (`data/
  extracted_fish/`, the same directory `pattern_extractor` read from) and
  converts to a scale-invariant fraction before aggregating, rather than
  shipping a resolution-confounded feature into a distance matrix.

**Distance metric.** Euclidean distance on rank-standardized per-species
feature vectors (`distance.py`) - Euclidean as the standard default for a
feature set mixing proportions, counts, and ratio-scale magnitudes with no
natural compositional (sum-to-one) structure that would motivate
Bray-Curtis instead; ranked (not raw z-scored) after the real 49-species
run showed why it matters, not speculatively - a single extreme outlier
species (Acanthurus lineatus's stripe-region count) compressed every other
species' raw z-score toward zero, distorting the stripe-dimension distance
matrix so a second genuinely-striped species (Zebrasoma veliferum) landed
closer to solid-coloured species than to A. lineatus. See
`standardize()`'s own docstring for the full before/after numbers. Kmult's
own admissibility requirements (log-ratio transforms for compositional
features, condition-number checks) are a Phase 4 concern for the raw
per-species feature matrices, not this phase's pairwise distances.

**Phylogeny.** `phylogeny.py` loads and prunes the same genetic-data-only
tree Phase 4 will use (`data/phylogeny/actinopt_12k_treePL.tre`), with the
tip-count range assertion and prune-before-cophenetic ordering the README's
Planned Approach step 4 specifies - built once here as the shared utility
both this phase's patristic matrix and Phase 4's Kmult input need, rather
than duplicated later.
"""
