# Methods

Full design rationale for the Surgeonfish Visual Phenomics and Phylogenetic
Inference pipeline. See [README.md](README.md) for results and how to run it,
[CHANGELOG.md](CHANGELOG.md) for the development history,
[BACKGROUND.md](BACKGROUND.md) for the pattern-genetics literature review, and
[`figures/`](figures/) for generated figures of each stage and result.

Much of what follows is **preregistration**: statistical choices were pinned
in writing, with reasons, before the tests were run. That matters here for a
specific reason — this project is a rebuild, and the pipeline it replaced
produced a phylogenetic-signal result that came from data-dredged feature
selection rather than a stated test. Where a choice was later found to be
wrong and changed, the change and its evidence are recorded rather than
quietly overwritten.

---

## 1. Fish identification and extraction

**Package:** `src/fish_extractor/` · **Notebook:** `notebooks/Phase1_Fish_Extraction.ipynb`

Grounded SAM 2 — Grounding DINO for zero-shot, text-prompted detection
(prompt: `"fish."`), then SAM 2.1 to turn the chosen box into a precise mask.
Both are Apache-2.0 and need no labelled training data, which matters because
~1,460 images across 64 species is far too little to train a detector from
scratch well.

A QA gate decides whether an image shows exactly one, roughly-centred fish:
box count after deduplication, box-centre offset from image centre, and box
area as a fraction of the image. Anything failing any criterion is **flagged
for human review**, never silently dropped or force-accepted.

Two real QA-gate bugs were found from actual box-coordinate data rather than
guessed (see CHANGELOG v1.4.1 / v1.4.2): duplicate detections of the same
fish were being counted as multiple fish (fixed with IoU-based deduplication),
and small background clutter boxes were doing the same (fixed by dropping
boxes under 20% of the largest qualifying box's area).

**Result:** 1,460 images processed, **856 accepted, 604 excluded, 0 left
flagged**, over three human review rounds (521 + 80 + 4 exclusions, recorded in
`reports/fish_extraction_review_feedback_round{1,2,3}.json`). Exactly one of
those 605 exclusions was later restored by human override
(`Acanthurus_achilles/021`, logged as `human_confirmed`), giving 604 net.

Note that `reports/fish_extraction_log.csv` is an append-only **event** trail —
1,770 rows for 1,460 unique images, since images were re-processed across
rounds. Read its per-key final status, not its raw row counts.

---

## 2. Pattern extraction

**Package:** `src/pattern_extractor/` · **Notebook:** `notebooks/Phase2_Pattern_Extraction.ipynb`

Three independent extractors run on each extracted fish crop, kept separate
rather than blended into one feature vector. That separation is not just an
engineering convenience: fish carry six distinct chromatophore types, and the
stripe and spot candidate-gene sets from Podobnik et al. (2020) are nearly
disjoint — stripe geometry tracks channel function (*kcnj13*), spotting tracks
gap-junction/adhesion genes (*gja4*, *gja5b*, *igsf11*), sharing only *gja5b*.
See [BACKGROUND.md](BACKGROUND.md).

| extractor | method | key outputs |
| --- | --- | --- |
| colour | reference-initialised k-means in a lightness-invariant hue/saturation space | `hue_dispersion`, `dominant_fraction`, `n_significant_colors`, `is_solid` |
| stripe | region elongation + width, per-channel FFT periodicity along body axes | `elongated_region_count`, `periodicity_strength`, `stripe_present` |
| spot | connected-component blob shape, size-capped | `spot_count`, `spot_area`, `spot_present` |

### Relationship to *patternize*

The colour-clustering step is a Python reimplementation of *patternize*'s
reference-initialised k-means method (Van Belleghem et al. 2018), the closest
published, peer-reviewed method for this task — demonstrated on fish and used
for phylogenetic comparative colour-pattern studies.

Two deliberate departures, both because this study compares 64 *different
species* rather than variation within one:

1. **No cross-specimen pixel registration.** *patternize*'s homology step
   assumes a broadly consistent body plan across specimens.
2. **The stripe/spot geometry analysis is this project's own extension** on
   top of the colour-region output, not a capability *patternize* provides.

**Validated against the R package.** Both implementations were run on identical
pixel sets from identical starting centres (`kImage(..., startCenter = ...)`),
giving a worst disagreement of **0.0037** across 12 cluster fractions on three
species — against a 0.05 threshold set before the run. See
`outputs/patternize_check/` and README.md.

Worth recording how the check itself had to be fixed: a first version compared
the two on the same *crop* and reported differences of 0.11–0.26. That was an
artifact, not a finding — `kImage()` clusters the entire raster including the
mask cutout's grey background, while `pattern_extractor` clusters masked-in
pixels only, so the two were partitioning different pixel sets. Rebuilding the
inputs to contain only masked-in pixels dropped the worst disagreement from
0.2638 to 0.0037 without touching the code under test.

### Validation against hand labels

`pattern_extractor` has no learned parameters, so the original plan's
"60/20/20 split" framing didn't apply — what it needed was a correctness
check on the feature extraction itself. A 158-image sample was hand-labelled
one image at a time (solid / striped / spotted); 3 were removed from the
dataset entirely as invalid inputs regardless of pattern (a fin-only crop, a
blown-out photo, a dried museum specimen), leaving **155 labels**.

That comparison found the extractors were performing *worse than trivial
baselines* and drove a full redesign (CHANGELOG v2.2.1–v2.2.3). Final agreement
against real hand labels on real SAM 2 masks:

| feature | agreement | precision | recall | F1 | TP |
| --- | --- | --- | --- | --- | --- |
| `is_solid` | 70.3% | 71.2% | 84.9% | 0.775 | 79 |
| `stripe_present` | 77.4% | 40.0% | 50.0% | 0.444 | 14 |
| `spot_present` | 80.0% | **10.5%** | **12.5%** | **0.114** | **2** |

Read F1, not agreement — agreement is inflated by class imbalance, which is why
`spot_present` scores highest on it while being the worst detector by a wide
margin (2 true positives against 17 false positives). `spot_present` should be
treated as unusable on its own; the spot dimension's signal comes from the
continuous features, not the boolean.

**Stripe thresholds were recalibrated in v6.2.0, overturning an earlier
conclusion.** The v2.2.1 calibration ran against masks *approximated* by a corner
flood-fill and carried an explicit caveat that production accuracy needed
re-checking; v2.2.3 then recorded that recall could not be raised without
collapsing precision, and declined an F2-weighted re-tune on that basis.

Re-running the same grid against the **real SAM 2 masks** shows that conclusion
was an artifact of the approximation:

| thresholds (eccentricity / count / width) | precision | recall | F1 |
| --- | --- | --- | --- |
| 0.97 / 20 / 0.12 (v2.2.1) | 40.0% | 14.3% | 0.211 |
| **0.98 / 8 / 0.10 (current)** | 40.0% | **50.0%** | **0.444** |

Recall triples at *identical* precision. Approximated masks generate far more
spurious elongated regions than real ones, which is why a count threshold of 20
looked necessary and is in fact far too high.

The chosen values are the **consensus of 200 stratified split-half trials**
(thresholds selected on one half, scored on the held-out half), not the single
F1-maximising point on the full set — selecting and evaluating on the same data
is the failure mode this pipeline exists to avoid. Held-out F1 improves 0.200 →
0.336, with the re-tuned settings winning in 93.5% of trials.

**Still read `stripe_present` cautiously.** Precision above ~50% is unreachable
at any threshold combination in the grid, so roughly half its positives are
wrong, and half of real stripe patterns are still missed. It is a
moderate-confidence signal in both directions — much better than the ~14%-recall
version it replaced, but not a confident one.

**Periodicity is confirmed dead**, not merely unvalidated. Re-checked on real
masks, its correlation with the hand labels is r=+0.020 / −0.092 / −0.066 at
`min_cycles` 2/3/4 — indistinguishable from noise, with two of three pointing the
wrong way. It is kept disabled rather than deleted so the mechanism and its
negative result stay legible.

---

## 3. Per-species aggregation and distance matrices

**Package:** `src/distance_matrices/` · **Notebook:** `notebooks/Phase3_Distance_Matrices.ipynb`

Collapses per-image feature rows into one vector per species, then builds
three species × species pattern-distance matrices plus one patristic-distance
matrix from the pruned reference tree.

**Aggregation rules:**

- **Boolean features aggregate as the proportion of a species' images scoring
  `True`**, not a majority vote — this preserves more information.
- **Numeric features use the arithmetic mean.** The original plan flagged
  circular-quantity handling (hue) as a concern; once the feature set was
  actually implemented, *every* exported value turned out to be a non-circular
  scalar, so a plain mean is correct throughout. A plan-vs-reality correction,
  not an oversight.
- **`mean_spot_area` is normalised before averaging.** `fish_extractor`
  deliberately doesn't resize crops, so raw pixel area isn't comparable across
  images of different native resolutions. Each image's area is converted to a
  fraction of its own masked-pixel count first. This was a real
  measurement-validity gap the original plan had *not* anticipated.

**Distance metric:** Euclidean on **rank-standardised** features.

Rank-transforming each column before z-scoring (rather than z-scoring raw
values) is a robust-statistics fix for a real bug found by the spot-check this
stage's verification criteria require. *Acanthurus lineatus*'s
`mean_elongated_region_count` (33.3) is such an extreme outlier that raw
z-scoring compressed every other species toward zero, burying genuinely
striped *Zebrasoma veliferum* (12.6 — 5th of 49) in the bulk and placing it
*closer to solid-coloured species* than to the other striped one. Rank
standardisation dropped that pair's distance from 6.4 to **0.9**. See
CHANGELOG v3.1.1.

**Analysis set size.** Two different counts get quoted in this project and
they describe different sets — worth separating carefully, since conflating
them is easy:

| set | source file | species | images | mean | median | range |
| --- | --- | --- | --- | --- | --- | --- |
| phylogeny-matched, reference photos **included** | `data/phylogeny/species_image_counts.csv` | 50 | 700 | 14.0 | 15 | 1–22 |
| **primary analysis set**, reference photos **excluded** | `reports/species_features.csv` | **49** | **648** | **13.2** | 14 | **2–21** |

Nothing reached the original 25-image target — Phase 1's QA gate and human
review thinned every species, so this is a continuous spread rather than a few
outliers against a full baseline.

*Naso maculatus* has exactly one image, and that image *is* its curated
reference photo (GBIF yielded zero usable field photos for the species). Since
`AggregationConfig.include_reference_images` defaults to `False`, it drops out
entirely, making the default analysis set **49 species, not 50** — handled
explicitly in code and asserted, not silently absorbed.

The two sparsest remaining species are *Naso tuberosus* (**2** images) and
*Acanthurus triostegus* (**4**) — and those are exactly the two dropped by the
`min_images_per_species=5` sensitivity run described in §4, verified directly
against `reports/species_features_min5.csv`. (*A. triostegus* reached the full
25/25 target in Phase 0 and lost most of its images to Phase 1's review — a
finding that only appeared once the real run happened.)

---

## 4. Phylogenetic signal testing

**Package:** `src/phylo_comparison/` · **Script:** `src/r/phase4_kmult.R` ·
**Notebook:** `notebooks/Phase4_Phylogenetic_Comparison.ipynb`

### 4.1 Why Kmult is primary, and Mantel only secondary

An earlier plan used Mantel tests as the primary analysis, with Harmon &
Glor's (2010) phylogenetic-permutation remedy cited as justification. Checked
against that paper's actual abstract, the remedy specifically targets inflated
type-I error in **three-way** Mantel tests. This project's per-dimension tests
are **two-way** (pattern distance vs. phylogenetic distance) — a case the same
paper says is better served by a **K-statistic** than by Mantel.

**Kmult** (Adams 2014) is the multivariate generalisation of exactly that
statistic, and takes per-species multivariate feature data directly, with no
reduction to a distance matrix. It is the primary test.

### 4.2 `physignal.z()`, not `physignal()`

Checked directly against the `geomorph` reference manual: `physignal()`'s own
documentation states that as of geomorph 4.0.2 it "no longer reports an effect
size" and defers to `physignal.z`. `physignal.z()`'s return value is a strict
superset — the K statistic, the optimised branch-scaling parameter (lambda),
and the standardised effect size (Z), all computed under **one** model rather
than as two analyses that could disagree with no principled way to reconcile
them.

### 4.3 Preregistered settings

| setting | value | reason |
| --- | --- | --- |
| `lambda` | `"front"` | see below |
| `PAC.no` | each dimension's full feature count (3–4) | already far below n=49; no reduction needed |
| `seed` | `NULL` | `physignal.z`'s deterministic default — identical P-values on repeated runs |
| correction | Benjamini–Hochberg across the 3 dimensions | applied to **both** primary and secondary tests |

**Why `lambda = "front"`** — and specifically not the alternatives:

- `"burn"` samples lambda to **maximise the effect size Z** rather than the
  log-likelihood. But Z is exactly what `compare.physignal.z` then compares
  across dimensions. Comparing three separately-maximised quantities is the
  shape of risk that produced this project's original data-dredged result, so
  it was designed away rather than merely noted.
- `"all"` tends to optimise lambda at 0 or 1, which is the documented `NaN`-Z
  failure case — the highest exposure to a known failure mode.
- `"front"`'s own documented cost is carried forward as a real limitation: it
  biases lambda toward 1, making the analysis more similar to raw multivariate K.

This matches the `geomorph` manual's own `compare.physignal.z` worked example,
which calls `physignal.z(..., lambda = "front", PAC.no = 7)` identically for
both traits before comparing them — the package authors' demonstrated usage
for precisely this "compare signal across trait subsets on one tree" case.

### 4.4 Cross-dimension comparison

The research question is comparative — *which* pattern dimension carries the
most signal — and three separate p-values cannot answer that, since K values
from different feature matrices aren't on a comparable scale.
`compare.physignal.z` (Collyer, Baken & Adams 2022) is the tool for it, run on
the three dimensions' `physignal.z` result objects.

**Why not `physignal.eigen`:** it decomposes *where within one feature matrix*
signal concentrates — a different question from comparing signal *across*
three separate matrices. Considered and set aside; potentially a useful
follow-up if one dimension turns out to dominate.

### 4.5 Data admissibility, not routine hygiene

Every `physignal*` argument in the `geomorph` manual is documented in terms of
**Procrustes shape variables**. The package's default assumption is shape data;
Adams (2014)'s title phrase ("shape and *other* high-dimensional multivariate
data") is the sole basis for applying it to non-shape colour/stripe/spot
features. That makes the transforms below **admissibility requirements for
using a shape-signal method on non-shape data** — skipping them would be
applying the method outside where its documentation shows it works.

Implemented in `src/phylo_comparison/feature_prep.py`:

- **Proportions-of-count** (`prop_solid`, `prop_striped`, `prop_spotted`) get
  the Smithson & Verkuilen (2006) boundary adjustment `(y(n−1) + 0.5) / n`,
  using each species' real image count as the trial count `n`, then a logit
  transform. Several species genuinely score 0 or 1, which a raw `logit` would
  send to ±∞.
- **Everything else** (means of continuous measurements, counts, magnitudes)
  gets `log1p` — deliberately *not* the Bernoulli-specific SV correction, since
  applying a proportion-of-count method to non-count data would repeat the
  exact "method used outside its documented scope" mistake this pipeline was
  rebuilt to avoid.
- Each matrix is then **z-score standardised**, and a **condition-number check**
  (threshold `1e10`) fails loudly rather than handing R a numerically unstable
  input.

**The compositional-data (CLR/ILR) concern in the original plan does not
apply.** No exported feature is a true sum-to-one composition — `color.py`
exports a scalar `dominant_fraction`, not a full cluster-fraction vector — so
there is nothing to log-ratio-transform.

### 4.6 Integrity checks

- **Tree identity:** the loaded tree's tip count must fall in `10,000 < n < 15,000`
  — a range chosen to discriminate the 11,638-tip genetic-data tree from the
  31,526-tip "complete" tree, rather than an exact `== 11638` match that would
  break on any release differing by one tip.
- **Prune before computing distances**, always: the unpruned patristic matrix
  is ~1.08 GB as float64 versus ~20 KB pruned.
- **Name-based joining:** `export_pruned_tree()` renames tree tip labels to each
  species' canonical key before writing the Newick file, so R joins feature rows
  to tips by name.
- **Exact set equality asserted in R** before anything runs — every feature
  matrix's species set must equal the tree's tip set, failing loudly rather
  than letting `physignal.z` silently drop or misalign rows.
- **Toolchain smoke test:** the script first runs `physignal.z` on `geomorph`'s
  own bundled `plethspecies` data and **hard-stops** unless K lands in 0.8–1.0
  (it returns 0.8901, matching the ~0.90 documented for that dataset). Without
  this, a low K on the real data would be uninterpretable — broken install or
  genuinely weak signal?

### 4.7 Secondary test: standard Mantel

The three pattern-distance matrices from Stage 3 are tested against the
patristic matrix with a **standard label-permutation Mantel test** (1,000
permutations, two-tailed).

A first implementation followed the original plan's phylogenetically-permuted
null, simulating traits under Brownian motion on the real tree. Run for real,
it returned near-1.0 p-values for *every* dimension regardless of signal
already confirmed by Kmult — a degenerate result. The cause was structural,
not a coding error: **BM trait divergence is by definition proportional to
patristic distance**, so a BM-simulated null sits near the theoretical
*ceiling* of phylogenetic correlation, not at "no association." Real, noisy
data will almost always look weaker, biasing every dimension toward
non-significance.

Standard label permutation is the legitimate null for this two-way,
fixed-reference case, and carries the ordinary Mantel caveats only — not the
three-way inflation risk that doesn't apply to this design. The two-tailed
criterion (`|null| ≥ |observed|`) is more conservative than the common
one-tailed "greater" default (e.g. `vegan::mantel`). See CHANGELOG v4.2.1.

### 4.8 Interpretation rules, fixed in advance

- A **null result is inconclusive, not evidence of no association** — Harmon &
  Glor found real power limitations even with the correct remedy applied.
- Where the primary and secondary tests disagree, **the primary test governs.**
  This rule did real work: spot was significant under Mantel (p=0.036) and not
  under Kmult (p=0.066), and the sensitivity re-run then showed the Mantel
  result was fragile, disappearing when the two sparsest species were dropped.

---

## 5. Phylogeny source and species matching

The reference tree is the Fish Tree of Life's **genetic-data-only** tree
(Rabosky et al. 2018), *not* its "complete" tree — the latter places species
lacking genetic data by stochastic polytomy resolution, a taxonomy-conditioned
random draw that its own documentation says should not be used for
trait-evolution analyses.

**50 of the 64 study species (78.1%)** have real genetic-data placement: 49 by
exact name match, plus *Zebrasoma veliferum* via a confirmed synonym (tree tip
*Zebrasoma velifer*). That synonym is currently cross-checked only against
NCBI's table, which is a sequence-database index rather than a fish taxonomic
authority — validating it against FishBase or Eschmeyer's Catalog of Fishes is
an open follow-up.

The 14 unmatched species are concentrated in *Acanthurus* (10 of 14) rather
than compounding the *Naso*/*Prionurus* sampling gaps. Most have GenBank
sequence data that simply wasn't part of this tree's fitted supermatrix — not
an absence of genetic data altogether.

Full audit trail: `data/phylogeny/species_coverage.csv` (with a `match_method`
column), `data/phylogeny/acanthuridae_synonyms.json`, and
`data/phylogeny/README.md`.

---

## References

- Adams, D. C. (2014). A generalized K statistic for estimating phylogenetic
  signal from shape and other high-dimensional multivariate data.
  *Systematic Biology*, 63(5), 685–697. https://doi.org/10.1093/sysbio/syu030
- Collyer, M. L., Baken, E. K., & Adams, D. C. (2022). A standardized effect
  size for evaluating and comparing the strength of phylogenetic signal.
  *Methods in Ecology and Evolution*, 13(2), 367–382.
  https://doi.org/10.1111/2041-210X.13749
- Harmon, L. J., & Glor, R. E. (2010). Poor statistical performance of the
  Mantel test in phylogenetic comparative analyses. *Evolution*, 64(7),
  2173–2178. https://doi.org/10.1111/j.1558-5646.2010.00973.x
- Rabosky, D. L., et al. (2018). An inverse latitudinal gradient in speciation
  rate for marine fishes. *Nature*, 559, 392–395.
  https://doi.org/10.1038/s41586-018-0273-1
- Smithson, M., & Verkuilen, J. (2006). A better lemon squeezer?
  Maximum-likelihood regression with beta-distributed dependent variables.
  *Psychological Methods*, 11(1), 54–71.
  https://doi.org/10.1037/1082-989X.11.1.54
- Sorenson, L., Santini, F., Carnevale, G., & Alfaro, M. E. (2013). A
  multi-locus timetree of surgeonfishes (Acanthuridae, Percomorpha), with
  revised family taxonomy. *Molecular Phylogenetics and Evolution*, 68(1),
  150–160. https://doi.org/10.1016/j.ympev.2013.03.014
- Van Belleghem, S. M., Papa, R., Ortiz-Zuazaga, H., Hendrickx, F., Jiggins,
  C. D., McMillan, W. O., & Counterman, B. A. (2018). patternize: An R package
  for quantifying colour pattern variation. *Methods in Ecology and Evolution*,
  9(2), 390–398. https://doi.org/10.1111/2041-210X.12853

Pattern-genetics references (Podobnik et al. 2020 and others) are cited in
[BACKGROUND.md](BACKGROUND.md).
