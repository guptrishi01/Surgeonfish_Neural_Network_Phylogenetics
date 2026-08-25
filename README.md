# Surgeonfish Visual Phenomics and Phylogenetic Inference

Do closely related surgeonfishes look more alike than distant ones? This
pipeline extracts colour, stripe, and spot pattern features from ~1,500
Creative-Commons photographs of 64 Acanthuridae species, reduces them to
per-species feature vectors, and tests each pattern dimension for
phylogenetic signal against a molecular phylogeny.

**Version 6.1.1** — all planned phases (0–5) complete and audited. See
[CHANGELOG.md](CHANGELOG.md) for the full history, [METHODS.md](METHODS.md) for
the statistical design, and [BACKGROUND.md](BACKGROUND.md) for the
pattern-genetics literature review.

---

## Research question

**Is visual pattern similarity among surgeonfish (family Acanthuridae)
associated with evolutionary relatedness — or does it arise independently of
shared ancestry?**

For 64 Acanthuridae species with a known molecular phylogeny, does quantified
visual similarity between species pairs covary with phylogenetic distance?

**What this design cannot distinguish.** Pattern covarying with phylogeny is
consistent with pattern being heritable, but equally consistent with pattern
tracking an ecological variable (diet, depth, reef zone) that is itself
phylogenetically conserved — closely related surgeonfishes tend to share
ecology, not just genes. Separating those would need ecological covariates and
a phylogenetic-regression approach (`procD.pgls`), which is out of scope here.
This is a correlational test; neither statistic identifies a genetic or
developmental mechanism.

---

## Results

Across **49 species** with real genetic-data phylogenetic placement
([`reports/species_features.csv`](reports/species_features.csv)):

| dimension | Kmult K | Kmult Z | Kmult BH *p* | Mantel *r* | Mantel BH *p* | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| **colour** | 0.0062 | 2.20 | **0.027** ✅ | +0.134 | **0.027** ✅ | **signal detected** |
| spot | 0.0362 | 1.71 | 0.066 | +0.100 | 0.036 | not significant (primary) |
| stripe | 0.0099 | −1.32 | 0.378 | −0.026 | 0.568 | null |

*Primary test: `geomorph::physignal.z` — [`outputs/phase4/kmult_results.csv`](outputs/phase4/kmult_results.csv).
Secondary: label-permutation Mantel — [`outputs/phase4/mantel_results.csv`](outputs/phase4/mantel_results.csv).
Benjamini–Hochberg corrected across the three dimensions.*

**Colour pattern shows weak but robust phylogenetic signal. Stripe and spot do not.**

### Robustness

Re-running everything with the two sparsest species dropped (49 → 47;
[`outputs/phase4/sensitivity_comparison.csv`](outputs/phase4/sensitivity_comparison.csv)):

| dimension | Kmult 49 → 47 | Mantel 49 → 47 | verdict stable? |
| --- | --- | --- | --- |
| colour | 0.027 → **0.018** | 0.027 → **0.045** | ✅ both |
| stripe | 0.378 → 0.379 | 0.568 → 0.621 | ✅ both |
| spot | 0.066 → 0.135 | 0.036 → **0.119** | ✅ primary / ❌ secondary |

**Every primary-test verdict is stable.** Colour stays significant and in fact
strengthens. The one flip is spot's *secondary* result, which loses
significance when *Naso tuberosus* (2 images) and *Acanthurus triostegus* (4)
are removed — direct evidence that spot's apparent Mantel significance rested
on two sparse species. Because the preregistered plan makes Kmult primary,
spot was already being reported as non-significant, so no headline conclusion
depends on it. Had the significant secondary result been reported instead, this
check would have shown that claim to be an artifact.

Cross-dimension comparison (`compare.physignal.z`) agrees: colour and spot each
carry more signal than stripe (*p*=0.014, 0.033) but are indistinguishable from
each other (*p*=0.714).

### How to read these numbers

1. **The effects are weak.** K ≈ 0.006 and *r* ≈ 0.13 for colour. The honest
   claim is "detectable but small," not "pattern tracks phylogeny closely."
2. **K is not a fraction of Brownian motion here.** Features pass through
   logit/`log1p` before testing; K is invariant to the linear standardisation
   that follows but **not** to those nonlinear transforms. Read K
   comparatively — across dimensions and species sets — not against K=1.
3. **Stripe's null is doubly inconclusive.** The stripe detector itself has
   only ~14% recall, so a null cannot distinguish "no phylogenetic signal in
   stripes" from "stripes weren't measured well enough." This is the clearest
   target for future work.
4. **Null results are inconclusive, not evidence of no association** (Harmon &
   Glor 2010).

---

## Pipeline

```
data/raw_images/            64 species · 1,460 CC-licensed photos (GBIF)
        │
        │  Phase 1 · src/fish_extractor/          [GPU · Colab]
        │  Grounding DINO detection → SAM 2.1 segmentation → QA gate
        │  856 accepted · 604 excluded · 0 left flagged
        ▼
data/extracted_fish/        mask-cutout crops, native resolution
        │
        │  Phase 2 · src/pattern_extractor/
        │  ┌─────────────┬─────────────┬─────────────┐
        │  │   colour    │   stripe    │    spot     │   3 independent
        │  │  k-means    │  elongation │    blob     │   extractors
        │  │ hue/sat spc │  + FFT      │   shape     │
        │  └─────────────┴─────────────┴─────────────┘
        │  validated against 155 hand labels: 70% / 81% / 80%
        ▼
reports/pattern_features.csv        853 rows, one per image
        │
        │  Phase 3 · src/distance_matrices/
        │  aggregate per species → rank-standardise → Euclidean
        ▼
reports/species_features.csv        49 species × 10 features
outputs/*_distance_matrix.csv       3 pattern + 1 patristic (49×49)
        │
        │  Phase 4 · src/phylo_comparison/ + r/phase4_kmult.R   [R · geomorph]
        │  Kmult (primary) + Mantel (secondary) + BH correction
        ▼
outputs/phase4/kmult_results.csv · mantel_results.csv · sensitivity_comparison.csv
```

Each phase was verified against its own real output before the next began —
see [CHANGELOG.md](CHANGELOG.md). Several phases had to be corrected that way:
Phase 2's first run flagged 99% of images as striped regardless of species, and
Phase 3's first run placed a striped species closer to solid-coloured ones than
to another striped one. Both were real bugs, found by checking output against
something known rather than by trusting the aggregate.

**Reconciling the counts**, since several look inconsistent at a glance — all
of these are checkable against the tracked files:

| quantity | value | source |
| --- | --- | --- |
| source images | 1,460 | `data/raw_images/` |
| log rows | 1,770 | append-only *event* trail — images were re-processed across review rounds |
| exclusion decisions | 605 | three non-overlapping review rounds (521 + 80 + 4), `reports/fish_extraction_review_feedback_round{1,2,3}.json` |
| later restored | −1 | `Acanthurus_achilles/021`, re-accepted as `human_confirmed` |
| **excluded** | **604** | 605 − 1 |
| **accepted** | **856** | 1,460 − 604 ✓ matches the log's accepted tally exactly |
| pattern rows | **853** | 856 − 3 images removed at validation as invalid inputs regardless of pattern (a fin-only crop, a blown-out photo, a dried museum specimen) |

---

## Running it

GPU-dependent stages run on **Google Colab**; everything else runs locally on
CPU. All pipeline code is plain importable Python, so the same functions work
from a script or a notebook cell.

```bash
git clone https://github.com/guptrishi01/Surgeonfish_Neural_Network_Phylogenetics.git
cd Surgeonfish_Neural_Network_Phylogenetics
pip install -e .            # base: requests, Pillow, numpy, scipy, biopython
pip install -e ".[vision]"  # adds torch/transformers/sam2 - only for Phase 1
pytest                      # 206 tests, no GPU needed
```

| phase | notebook | runtime | notes |
| --- | --- | --- | --- |
| 0 · collect images | `python -m dataset_builder.cli` | CPU, local | rate-limited GBIF sourcing |
| 1 · extract fish | [`Phase1_Fish_Extraction.ipynb`](notebooks/Phase1_Fish_Extraction.ipynb) | **GPU** (L4) | needs `[vision]` extra |
| 2 · extract patterns | [`Phase2_Pattern_Extraction.ipynb`](notebooks/Phase2_Pattern_Extraction.ipynb) | CPU | pure NumPy/SciPy/Pillow |
| 3 · distance matrices | [`Phase3_Distance_Matrices.ipynb`](notebooks/Phase3_Distance_Matrices.ipynb) | CPU | fast, deterministic |
| 4 · phylogenetic tests | [`Phase4_Phylogenetic_Comparison.ipynb`](notebooks/Phase4_Phylogenetic_Comparison.ipynb) | CPU + **R** | `rpy2` + `geomorph` |

The Colab notebooks clone the project **onto Google Drive** rather than
ephemeral local disk — that's what keeps resumable per-image state, extracted
images, and review pages valid across a disconnect or restart.

Phase 4 installs `geomorph` from source on first run (5–15 min) and executes
`r/phase4_kmult.R` via `rpy2`'s `%%R` magic, so one Python-kernel session
handles both halves without a runtime switch.

---

## Species and phylogeny

64 species across all six extant Acanthuridae genera: *Acanthurus* (31),
*Naso* (14), *Ctenochaetus* (9), *Zebrasoma* (6), *Prionurus* (3),
*Paracanthurus* (1).

**Taxon sampling is uneven.** Against each genus's valid-binomial species count
in NCBI Taxonomy (hybrids and `sp.`/`aff.`/`cf.`/BOLD placeholders excluded —
see `data/genus_species_counts.json`): *Prionurus* 43%, *Naso* 70%,
*Acanthurus* 79%, *Zebrasoma* 86%, *Ctenochaetus* 100%, *Paracanthurus* 100%,
against a family mean of **64/83 (77%)**. Missing species were dropped for
image availability in Phase 0, not by design. *Prionurus* is the only genus
clearly below the family mean; *Naso* is second-lowest, which matters because
sparse sampling in one of Acanthuridae's two deep clades disproportionately
shapes the large phylogenetic distances that most influence these tests.

**Phylogeny.** The Fish Tree of Life's **genetic-data-only** tree (Rabosky et
al. 2018) — not its "complete" tree, which places species lacking genetic data
by stochastic polytomy resolution, something its own documentation says
shouldn't be used for trait-evolution analysis. **50 of 64 species (78.1%)**
have real genetic placement; the analysis set is 49 after *Naso maculatus*
drops out (its only image is its curated reference photo). Full audit trail in
[`data/phylogeny/`](data/phylogeny/).

**A testable prediction.** *Acanthurus* is paraphyletic with respect to
*Ctenochaetus*; Sorenson et al. (2013) recover this specifically as
*Ctenochaetus* nested with *A. nubilus* and *A. pyroferus*. Both named species
are present in the matched set, so whether that paraphyly also appears in
visual similarity is checkable as named. Caveat: restricting to genetically
sampled species drops *Acanthurus* coverage to **54%**, the lowest of any
genus post-restriction.

---

## Data sourcing and compliance

Images are collected programmatically via `src/dataset_builder/` — not
hand-collected, not scraped from web search results.

- **Source: [GBIF](https://www.gbif.org)'s REST API**, which aggregates
  research-grade CC-licensed photos from iNaturalist and others. iNaturalist's
  own API host disallows generic automated query-string access; going through
  GBIF respects that boundary rather than routing around it.
- **robots.txt is enforced at runtime** before every request — including each
  image download, which can land on a different host than the API.
- **License allowlist:** CC0/public-domain and CC-BY/BY-SA/BY-NC/BY-NC-SA only.
  No-derivatives licenses are excluded, since this pipeline produces derivative
  works. License and attribution are logged per image to
  [`reports/image_sourcing_log.csv`](reports/image_sourcing_log.csv).
- **Rate-limited and self-identifying:** 1 request/second, descriptive
  `User-Agent` with contact address, backoff on transient errors.
- **Human visual review**, not more heuristic tuning — automated filters catch
  mechanical problems but can't judge "is this a clean, close-up, single-fish
  photo."
- **Open action:** GBIF recommends registering a derived-dataset DOI for
  search-API pulls. Required before publication; not yet done.

---

## Repository structure

```
├── src/
│   ├── dataset_builder/       Phase 0 · GBIF sourcing, license/robots compliance
│   ├── fish_extractor/        Phase 1 · Grounded SAM 2 detection + segmentation
│   ├── pattern_extractor/     Phase 2 · colour/stripe/spot feature extraction
│   ├── distance_matrices/     Phase 3 · per-species aggregation + distances
│   └── phylo_comparison/      Phase 4 · Kmult feature prep + tree export
├── r/phase4_kmult.R           Phase 4 · physignal.z, compare.physignal.z, Mantel
├── notebooks/                 Colab notebooks, one per phase
├── tests/                     183 tests, GPU calls mocked
├── data/
│   ├── raw_images/            64 species · zipped source photos
│   ├── phylogeny/             reference tree, coverage table, synonyms
│   ├── genes/                 candidate genes from the literature review
│   └── genome_assemblies/     NCBI availability survey (metadata only)
├── reports/                   per-image features, audit logs, hand labels
├── outputs/                   distance matrices + Phase 4 results
├── METHODS.md                 statistical design and preregistration
├── CHANGELOG.md               full version history
└── BACKGROUND.md              pattern-genetics literature review
```

Result tables under `reports/` and `outputs/` are tracked deliberately — every
number above traces to one, so a clone can check them without re-running a GPU
pipeline.

---

## Limitations

- **Weak effect sizes.** Detectable, not strong. See "How to read these numbers."
- **Features within a dimension are partly redundant.** Each dimension carries
  both a continuous measure and a proportion thresholded from that same
  measure, so they correlate strongly by construction: `mean_hue_dispersion` ~
  `prop_solid` (*r*=−0.95), `mean_elongated_region_count` ~ `prop_striped`
  (*r*=+0.92), `mean_spot_count` ~ `prop_spotted` (*r*=+0.80). The matrices stay
  well-conditioned (condition numbers 7.5 / 5.4 / 3.2) so the tests are valid,
  but "4 colour features" overstates the independent information — the
  effective dimensionality is lower than the feature count suggests.
- **Stripe detection has ~14% recall.** `stripe_present=False` means "not
  confidently detected," not "confirmed absent." The stripe null is therefore
  inconclusive about biology.
- **Correlation, not mechanism.** These tests cannot separate inherited pattern
  from phylogenetically conserved ecology.
- **n=49.** Small for comparative methods; power is limited and null results
  should be read accordingly.
- **Uneven taxon sampling**, especially *Prionurus* (43%) and *Naso* (70%),
  and *Acanthurus* at 54% after the phylogeny restriction.
- **The *patternize* port is unvalidated** against the original R package — no
  numerical-equivalence check exists yet.
- **The synonym table is NCBI-derived**, not a fish taxonomic authority. It
  gates the one match (*Zebrasoma veliferum*) that makes coverage 50 rather
  than 49.

---

## Open follow-ups

Distinct from the limitations above: those bound what the result means, these
are actionable and unfinished.

| # | item | why it matters |
| --- | --- | --- |
| 1 | **Register a GBIF derived-dataset DOI** | GBIF's own guidance requires this for search-API pulls. **Blocks publication** — the only hard blocker here. |
| 2 | **Improve `stripe_present` recall** (~14%) | The highest-value scientific next step. Stripe is the one dimension whose null is uninterpretable for a fixable reason. |
| 3 | **Numerical-equivalence check for the *patternize* port** | The colour clustering is a Python reimplementation never checked against the R package — and colour is the dimension carrying the headline result. |
| 4 | **Cross-check the synonym table** against FishBase or Eschmeyer's Catalog of Fishes | One species' inclusion currently rests on an NCBI-derived table that isn't a fish taxonomic authority. |
| 5 | **Consider dropping the redundant thresholded proportions** | `prop_solid`/`prop_striped`/`prop_spotted` duplicate their own continuous measures (\|r\| = 0.80–0.95). Re-running without them would show whether the result depends on that redundancy. |

Item 3 is worth flagging as the largest unquantified risk in the pipeline: it
sits directly upstream of the only significant finding.

---

## Status

Rebuilt from scratch after an earlier pipeline produced unreliable results — a
broken evaluation metric, a misreported ROC-AUC, a silently dropped test image,
and a phylogenetic-signal result that came from data-dredged feature selection
rather than a preregistered test. Every statistical choice here was pinned in
writing before the tests ran ([METHODS.md](METHODS.md)), and every phase was
verified against real output before the next began.

**Phase 5 (this documentation rewrite) was the last planned phase.**

### Phase 6 was scoped and deliberately not pursued

Phase 6 — checking whether the 56 candidate genes from
[BACKGROUND.md](BACKGROUND.md) appear in available genome assemblies — was
always contingent on Phase 4 finding significance. It technically did, for
colour. The phase was then scoped properly, and the evidence does not support
running it:

| constraint | value |
| --- | --- |
| species with a public genome assembly | 16 of 64 |
| …**also** in the 49-species analysis set | **14** |
| …at chromosome level (rest are scaffold) | **1** |
| Acanthuridae skin transcriptomes in the literature | **0** |

At n=14, a genotype–phenotype correlation would need **|r| ≥ 0.78** to survive
correction across the 52 colour candidate genes. Power to detect a true
*r*=0.5 — already strong for a polygenic trait — is **6.9%**. For scale, the
phenotype signal this would be chasing is *r*=0.13.

So a null result would be uninterpretable (no power to distinguish "no
association" from "couldn't have seen one"), while a positive result at n=14
across 52 genes would more likely be noise — which is precisely the
data-dredging failure mode this project was rebuilt to escape. The
presence/absence half is feasible but near-vacuous: teleost pigmentation genes
are deeply conserved, so finding them in a surgeonfish genome is close to
guaranteed and says nothing about pattern variation. The expression half has no
data to run on at all.

**The higher-value next step is measurement, not genomics:** fixing
`stripe_present`'s ~14% recall. Stripe is currently the one dimension where the
biological question is unanswerable for a fixable reason — the detector, not
nature.

---

## Acknowledgements

Conducted in the **Dornburg Lab** at the University of North Carolina
Charlotte. Phylogenetic reference data from the
[Fish Tree of Life](https://fishtreeoflife.org). Image data from
[GBIF](https://www.gbif.org) contributors under Creative Commons licenses.
GPU computation on Google Colab. The Phase 2 colour-clustering methodology is
a Python reimplementation of *patternize* (Van Belleghem et al. 2018); the
Phase 4 statistics run in *geomorph* (Adams, Collyer et al.). Full citations
in [METHODS.md](METHODS.md).
