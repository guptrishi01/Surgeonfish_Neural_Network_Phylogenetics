"""Phase 4 of the surgeonfish ML pipeline: phylogenetic-signal statistical tests.

Prepares Phase 3's per-species feature matrices (`reports/species_features.
csv`) and pruned tree for the primary Kmult test (`geomorph::physignal.z`
in R - see `r/phase4_kmult.R`) and the secondary phylogenetically-permuted
Mantel check, per the project README's Planned Approach step 4. This
package covers only the Python-side preparation - the statistical tests
themselves run in R, per that plan's explicit "not simple formulas to
port" decision (see README's 1.2.3/1.2.4 changelog entries): `physignal.z`
and `compare.physignal.z` involve phylogenetically-aligned component
projection, branch-scaling likelihood optimization, RRPP permutation, and
a Box-Cox effect size - an "almost right" hand port would return a
plausible, wrong number with no visible error, and this project has
already shipped one silently-wrong metric once.

**Plan-vs-actual reconciliation.** Like Phase 3's aggregation step (see
`distance_matrices`'s own docstring for the circular-hue and
`mean_spot_area` cases), this phase's admissibility requirements were
planned before the actual feature set was implemented, and don't map onto
it exactly:

- **No true compositional (sum-to-one) feature exists.** The plan
  anticipated colour-cluster fractions summing to 1, needing a CLR/ILR
  log-ratio transform. `pattern_extractor.color`'s actual output is a
  single `dominant_fraction` scalar (plus `hue_dispersion`,
  `n_significant_colors`), not the full k-length fraction vector - there
  is nothing to take a log-ratio *between*. CLR/ILR is not applied here,
  since applying a compositional-data transform to non-compositional data
  would itself be exactly the "using a method outside where its own
  documentation says it's been shown to work" mistake this project has
  spent five external audit rounds trying to avoid.
- **Two genuinely different kinds of [0, 1]-bounded feature exist**, and
  are treated differently rather than both forced through one generic
  fix: `prop_solid`/`prop_striped`/`prop_spotted` are true
  proportions-of-count (a fraction of a species' actual images scoring
  True), so they get the Smithson & Verkuilen (2006) boundary-avoidance
  adjustment - designed specifically for proportion data with genuine 0/1
  values, using each species' own `n_images` as the trial count - before
  a logit transform. `mean_dominant_fraction`, `mean_periodicity_
  strength`, and `mean_spot_area_fraction` are means of continuous
  per-image values, not proportions of successes out of trials - applying
  the same Bernoulli-specific correction to them would be a second
  instance of the CLR-on-non-compositional-data mistake, so they're
  log1p-transformed instead, the same as the count/magnitude features.
- **PAC.no reduction is explicitly not applied.** The plan (following the
  `geomorph` manual's own shape-data-with-many-landmarks framing) expected
  a PCA dimensionality-reduction step before `physignal.z`. Every actual
  pattern dimension here has only 3-4 raw features already, far below
  n=49 species - there is no high-dimensionality problem to reduce.
  `PAC.no` is set to each dimension's full feature count (no reduction),
  reported explicitly rather than defaulted silently, per the plan's own
  "not chosen post hoc" requirement.

See ``feature_prep.py`` for the transform implementations and
``export.py`` for what's written to disk for R to consume.
"""
