# Surgeonfish Visual Phenomics and Phylogenetic Inference

## Version

**1.2.2** — see [CLAUDE.md](CLAUDE.md) for the versioning scheme (major = phase completion, minor = a step within a phase, patch = a bug fix).

**Changelog**
- **1.2.2** — A second external audit round re-checked the v1.2.1 corrections (all confirmed exact against Crossref/NCBI) and found a real bug plus three scope-accuracy gaps, all independently re-verified before acting: (1) **Bug fix**: species-matching against the Fish Tree of Life used exact string matching only, missing a confirmed synonym (*Zebrasoma veliferum* = tree tip *Zebrasoma velifer*, same NCBI tax ID). Corrected count: **50/64 species (78.1%)**, not 49/64 — re-verified directly against the tree file with bidirectional synonym resolution; the other 17 of 18 "unused" tree tips turned out to be genuinely different Acanthuridae species not in this study, not recoverable matches, so the audit's implied scale of the bug was larger than what was actually fixable. `data/phylogeny/acanthuridae_synonyms.json` (the NCBI synonym table used for this check) and a tree-file checksum are now saved for reproducibility. (2) **Overreaching claim, corrected**: the Research Question implied the test could distinguish "shared ancestry" from "shared habitat/ecology" as the reason for pattern similarity — Kmult/Mantel can't make that distinction on their own, since ecology is itself phylogenetically structured; the claim is now scoped to what these statistics actually measure (pattern-phylogeny covariance), with the ecological-covariate/PGLS extension needed for the stronger claim noted as out of scope for now. (3) Added Kmult's feature-matrix assumptions (compositional-data handling, standardization, dimensionality vs. n=50) and `physignal.z`/`compare.physignal.z` (Collyer, Baken & Adams 2022) for cross-dimension effect-size comparison to the Phase 4 plan — not yet built, so this changed requirements, not code. (4) Extended the multiple-comparisons-correction requirement to the primary Kmult tests, not just the secondary Mantel tests. Also confirmed via NCBI Nucleotide that most of the still-unmatched 14 species have GenBank sequence data not incorporated into this particular Fish Tree of Life release, a more accurate framing than "no genetic data exists."
- **1.2.1** — Corrections from an external README audit, each independently re-verified against a primary source before acting on it: (1) the patternize citation had two fabricated author names ("Planas, S.", "Martin, S. H.") that don't exist in the real 7-author list — fixed everywhere it appeared, confirmed against Crossref. (2) The "nine molecular loci" phrase was misattributed to the Fish Tree of Life; it actually describes a different paper (Sorenson et al. 2013) — fixed, and that paper is now properly cited as the alternative phylogeny source considered. (3) The 1.0.0 entry's image counts (~1,850/~490) didn't reconcile against the actual dataset; replaced with the real numbers read directly from `data/raw_images_state.json` (2,329 evaluated, 1,460 accepted, 869 rejected — internally consistent with 55×25 + the 9 short species' counts). (4) **Blocking methodological finding, resolved**: the Fish Tree of Life's complete phylogeny places species lacking genetic data via stochastic polytomy resolution, which its own documentation says shouldn't be used for trait-evolution analyses (Rabosky 2015) — checked directly (`data/phylogeny/`, robots.txt-permitted download of the genetic-data-only tree): **49/64 study species (76.6%) have real genetic-data placement**; Phase 4 will restrict the phylogenetic side of the analysis to those 49 rather than use imputed placements for the other 15. (5) **Second blocking finding, resolved**: Harmon & Glor's (2010) phylogenetic-permutation remedy, previously cited to justify this project's Mantel tests, specifically addresses *three-way* Mantel tests — this project's per-dimension tests are two-way, a case the same paper says is better served by a K-statistic. Kmult (Adams 2014) is now the planned primary test per dimension, with the phylogenetically-permuted Mantel tests kept as a secondary check (see Planned Approach). (6) Documented taxon-sampling unevenness across genera (*Prionurus* 43%, *Naso* 58% vs. 60% family mean) and the *Ctenochaetus*/*Acanthurus* paraphyly's taxonomic-convention implications as explicit limitations (Species section). None of Phase 4's code exists yet, so this round changed plans and documentation, not implementation.
- **1.2.0** — Phase 1 (`src/fish_extractor/`) code-complete and unit-tested: Grounded SAM 2 zero-shot fish detection/segmentation, the accept/flag QA gate, resumable per-image state, a human-review page for flagged images, and a `--run`-gated CLI (mirrors `dataset_builder`'s `--scrape-web` pattern — no model loads unless explicitly asked). Not yet run against the real dataset — that needs Colab's GPU and is the actual Phase 1 completion event. Also added a working Phase 2 prototype (`src/pattern_extractor/`): coloring/stripe/spot feature extraction from `fish_extractor`'s output, built around a Python reimplementation of *patternize*'s (Van Belleghem et al. 2018) reference-initialized k-means colour-clustering method — see Planned Approach below for the full citation and how this differs from the original R package. Two rounds of independent code review caught and fixed real bugs before either lands: a missing file extension that would have silently skipped `.webp` images, a non-deterministic clustering seed, a stripe-periodicity check that was blind to same-luminance/different-hue stripes, an unguarded crash path on degenerate (near-solid-colour or empty-mask) images, and a dead config field. 106 tests passing.
- **1.1.0** — Literature review completed for the candidate-genetics/methodology groundwork behind the pattern-extraction and phylogenetic-comparison design (see [BACKGROUND.md](BACKGROUND.md)): 56 NCBI-validated candidate genes across the coloring/stripes/spots dimensions (`data/genes/`), sourced from five papers on zebrafish, spotted scat, medaka, and mammalian pigmentation genetics — developmental corroboration for treating the three pattern dimensions as separable, not just an engineering convenience. Also surfaced a Mantel-test methodology correction (Harmon & Glor 2010): phylogenetic permutation, not naive permutation, and a null result must be reported as inconclusive rather than as evidence of no association — both folded into the Phase 4 plan. Separately, queried NCBI for genome-assembly availability across the 64 species (`data/genome_assemblies/`): 16/64 (25%) have a public assembly. Metadata only, no sequence downloaded — this is groundwork for a later, contingent phase, not new pipeline code.
- **1.0.0** — Phase 0 (data collection) complete. 2,329 candidate images evaluated from GBIF across all 64 species over 9 human-review rounds; 1,460 accepted into the final dataset (869 rejected and, where possible, backfilled). 55/64 species reached the 25-image target (55×25 + the 9 short species' actual counts = 1,460, reconciled directly against `data/raw_images_state.json`); the remaining 9 are genuinely limited by GBIF's available licensed photos, not a pipeline gap. Along the way: fixed a Windows encoding crash, a filename-numbering bug (with a cleanup pass over the existing dataset), and confirmed automated close-up/background filtering isn't reliable enough to replace human review.

This section will be trimmed down once the project is finished — for now it's tracking everything as it happens.

---

## Research Question

**Is visual pattern similarity among surgeonfish (family Acanthuridae) genetically associated with evolutionary relatedness — or does it arise independently of shared ancestry?**

Concretely: for a set of Acanthuridae species with a known molecular phylogeny, does quantified visual similarity (color pattern, texture) between species pairs correlate with their phylogenetic distance? A significant result (tested primarily via Kmult, a multivariate phylogenetic-signal statistic, with a distance-based Mantel test as a secondary check — see Planned Approach) is evidence that visual pattern covaries with phylogenetic relatedness — closely related species tend to look more alike than distantly related ones do. A null result means visual pattern is not explained by relatedness at the level these methods can detect, and any resemblance between species is more likely convergent.

**What this test cannot, on its own, distinguish**: pattern covarying with phylogeny is consistent with pattern being heritable, but it's equally consistent with pattern tracking an ecological variable (diet, depth, reef zone) that is itself phylogenetically conserved — closely related surgeonfishes tend to share ecology, not just genes. Kmult/Mantel alone cannot separate "inherited" from "conserved-ecology-mediated" as the reason for a significant result. Testing that distinction specifically would require ecological covariates (e.g. from FishBase) and a phylogenetic-regression approach (`procD.pgls`) — not yet in scope, noted here so the claim this project can support isn't overstated.

This is a correlational test, not a claim of direct causation: both statistics establish association between visual pattern and phylogeny. Neither, on its own, identifies the underlying genetic or developmental mechanism, or separates heritability from conserved ecology.

---

## Status

This project is being rebuilt from scratch after an earlier pipeline run produced unreliable results (a broken evaluation metric, a misreported ROC-AUC, a silently-dropped test image, and a phylogenetic-signal result that came from data-dredged feature selection rather than a preregistered test). Rebuilding proceeds one phase at a time (see [CLAUDE.md](CLAUDE.md)), each verified against its own output before moving to the next — see the changelog above for what's done so far.

Nothing in this README describes final scientific results yet — that section gets written last, once every number in it traces back to a real output file.

---

## Species

64 species across all six extant Acanthuridae genera, sourced from lateral reference photographs:

| Genus | Species count |
|---|---|
| *Acanthurus* | 31 |
| *Ctenochaetus* | 9 |
| *Naso* | 14 |
| *Paracanthurus* | 1 |
| *Prionurus* | 3 |
| *Zebrasoma* | 6 |
| **Total** | **64** |

**Taxon sampling is uneven across genera** relative to each genus's full species count in NCBI Taxonomy: *Prionurus* 3/7 (43%), *Naso* 14/24 (58%), *Zebrasoma* 6/10 (60%), *Acanthurus* 31/52 (60%), *Ctenochaetus* 9/13 (69%), *Paracanthurus* 1/1 (100%) — against a family mean of 60%. This matters more than a generic coverage gap: *Naso* is one of Acanthuridae's two deep clades (Nasinae, crown age ~17 Ma per Sorenson et al. 2013 below), so sparse, non-random sampling there disproportionately shapes the large phylogenetic distances that most influence a Mantel/Kmult result. Missing species were dropped for image availability (Phase 0), not by design — this is a real limitation, not a curated subsample.

A key biological complication: *Acanthurus* is paraphyletic with respect to *Ctenochaetus* in the molecular phylogeny — several *Ctenochaetus* species are nested inside the *Acanthurus* clade. Whether that paraphyly is also reflected in visual similarity is one specific, checkable prediction of the broader research question above. Sorenson et al. (2013, cited below) recommend dissolving *Ctenochaetus* into *Acanthurus* on this basis; this project keeps them as separate genus labels (standard reference-database convention) since genus is a descriptive grouping here, not part of the statistical test itself, which operates at the species-tip level.

The molecular phylogeny used as the evolutionary-relatedness reference comes from the Fish Tree of Life (fishtreeoflife.org, Rabosky et al. 2018, *Nature* 559, 392–395), specifically its **genetic-data-only tree** (not the "complete" tree, which places species lacking genetic data via stochastic polytomy resolution — a taxonomy-conditioned random draw, not phylogenetic signal; see `data/phylogeny/README.md` for the full reasoning, source, and a checksum for reproducibility). **50 of the 64 study species (78.1%) have real genetic-data placement** in that tree (49 by exact name match, 1 — *Zebrasoma veliferum* — via a confirmed taxonomic synonym); the analysis is restricted to those 50. The 14 still unmatched are concentrated in *Acanthurus* (10 of 14; already the best-sampled genus) rather than compounding the *Naso*/*Prionurus* sampling gap above — most have GenBank sequence data that simply wasn't part of this tree's fitted supermatrix, not an absence of genetic data altogether; see `data/phylogeny/species_coverage.csv` and `README.md` for the full per-species breakdown and reasoning. ("Nine molecular loci" describes a different, Acanthuridae-specific timetree — Sorenson, L., Santini, F., Carnevale, G., & Alfaro, M. E. (2013). A multi-locus timetree of surgeonfishes (Acanthuridae, Percomorpha), with revised family taxonomy. *Molecular Phylogenetics and Evolution*, 68(1), 150–160. https://doi.org/10.1016/j.ympev.2013.03.014 — considered as the phylogeny source and not currently used; see `data/phylogeny/README.md`.)

---

## Data Sourcing & Compliance

Images are collected programmatically via `src/dataset_builder/` (see its package docstring for full detail), not hand-collected or scraped from arbitrary web search results. In summary:

- **Source**: [GBIF](https://www.gbif.org)'s public REST API (`occurrence/search`), which aggregates research-grade, Creative-Commons-licensed photos from iNaturalist and other providers. GBIF's own `robots.txt` permits this; iNaturalist's own API host disallows generic automated query-string access, so this pipeline goes through GBIF rather than around that boundary.
- **robots.txt is enforced at runtime** before every request this pipeline makes — including each image download, which can land on a different host than the API itself — not just checked once by hand.
- **License allowlist**: only CC0/public-domain and CC-BY/BY-SA/BY-NC/BY-NC-SA images are accepted; no-derivatives (ND) licenses are excluded since this pipeline produces derivative works (masks, crops, figures). License and attribution are logged per image to `reports/image_sourcing_log.csv`.
- **Rate-limited and self-identifying**: one request/second by default, with a descriptive `User-Agent` and contact address, and backoff-retry (not hammering) on transient server errors.
- **Human visual review**: automated filters catch mechanical problems (resolution, duplicates) but can't reliably judge "is this a clean, close-up, single-fish photo" — that judgment is made via a generated review page (`reports/review.html`), not by further heuristic tuning.
- **Citation follow-up (open)**: GBIF's own guidance recommends registering a "derived dataset" DOI for data pulled via the search API (as opposed to a bulk download) — this is required before this dataset is cited in any publication, and is tracked as an open action, not yet done.

Full detail, including the reasoning behind each of these choices, is in [CLAUDE.md](CLAUDE.md).

---

## Planned Approach

1. **Fish identification & extraction** (`src/fish_extractor/`, code-complete, pending a real Colab/GPU run) — Grounded SAM 2 (Grounding DINO zero-shot text-prompted detection + SAM 2.1 segmentation) identifies whether an image shows exactly one, roughly-centered fish and, if so, extracts it as a mask-cutout crop. No training required; anything ambiguous is routed to a human-reviewable page rather than silently accepted or dropped.
2. **Pattern extraction** (`src/pattern_extractor/`, working prototype) — three independent feature extractors run on the fish-extractor's output: coloring (dominant-colour clustering), spots/freckles (connected-component blob shape), stripes (region elongation + per-channel FFT periodicity along the body axes). Kept separate rather than one blended feature vector, because these are developmentally distinct pattern-generating mechanisms (see [BACKGROUND.md](BACKGROUND.md)) — corroborated by the candidate-gene literature review (`data/genes/`), not just an engineering choice. The colour-clustering step is a Python reimplementation of *patternize*'s reference-initialized k-means method (Van Belleghem, S. M., Papa, R., Ortiz-Zuazaga, H., Hendrickx, F., Jiggins, C. D., McMillan, W. O., & Counterman, B. A. (2018, online 2017). patternize: An R package for quantifying colour pattern variation. *Methods in Ecology and Evolution*, 9(2), 390–398. https://doi.org/10.1111/2041-210X.12853), the closest published, peer-reviewed method for this task — demonstrated on fish and used for phylogenetic comparative colour-pattern studies. Two deliberate departures from the original, both because this study compares 64 different species rather than variation within one: no cross-specimen pixel registration (patternize's homology step assumes a broadly consistent body plan), and the stripe/spot geometry analysis is this project's own extension on top of patternize's colour-region output, not a capability patternize itself provides.
3. **Distance matrices** — three separate species x species visual-distance matrices (one per pattern dimension), plus one patristic-distance matrix from the restricted (genetically-sampled) molecular tree, for the secondary Mantel check described below.
4. **Statistical test** — **Kmult** (Adams, D. C. (2014). A generalized K statistic for estimating phylogenetic signal from shape and other high-dimensional multivariate data. *Systematic Biology*, 63(5), 685–697. https://doi.org/10.1093/sysbio/syu030 — implemented in R as `geomorph::physignal()`/`physignal.z()`, reimplemented in Python here) is the primary per-dimension test, run on each pattern dimension's per-species multivariate feature data against the restricted 50-species phylogeny (`data/phylogeny/`). This replaces an earlier plan to use Mantel tests with Harmon & Glor's (2010) phylogenetic-permutation remedy as primary — that remedy specifically targets inflated type-I error in *three-way* Mantel tests, and this project's per-dimension tests are two-way (pattern vs. phylogeny), which Harmon & Glor's own paper says is better served by a K-statistic (Kmult is its multivariate generalization) than by Mantel. Comparing signal strength *across* the three dimensions (not just each against a null) needs effect sizes, not three separate p-values — `physignal.z`'s standardized effect size and `compare.physignal.z` (Collyer, Baken & Adams (2022). A standardized effect size for evaluating and comparing the strength of phylogenetic signal. *Methods in Ecology and Evolution*, 13(2), 367–382. https://doi.org/10.1111/2041-210X.13749) are the planned tools for that, both to be ported alongside Kmult itself. The three pairwise-distance Mantel tests (one per pattern dimension, phylogenetically-permuted) are retained as a **secondary**, distance-based check. A multiple-comparisons correction across the three dimensions applies to both the primary Kmult results and the secondary Mantel results before anything is called significant. Under either test, a null result is reported as inconclusive, not as evidence of no association — Harmon & Glor found the Mantel test has real power limitations even with the correct remedy applied. Feature-matrix requirements for Kmult (compositional-data handling, standardization, dimensionality relative to n=50) are tracked in the implementation plan, not yet built.
5. **(Contingent, deferred)** If any pattern dimension shows a significant, corrected result: investigate whether the 56 candidate genes identified in the literature review (`data/genes/`) are present/annotated in the genome assemblies available for 16 of the 64 species (`data/genome_assemblies/`), and whether any expression data can be found. Not yet scoped in detail.

Implementation details (exact feature set, split strategy for validating pattern extraction against manually labeled ground truth) are being finalized as each stage is rebuilt.

---

## Repository Structure

```
Surgeonfish_Neural_Network_Phylogenetics/
├── data/
│   ├── raw_images/                    # One subfolder per species, one per genus
│   │   ├── Acanthurus/
│   │   │   └── Acanthurus_guttatus/
│   │   │       ├── 000_reference.jpg  # Pre-existing curated photo (seed image)
│   │   │       └── 001_gbif_....jpg   # GBIF-sourced, license-checked, reviewed
│   │   ├── Ctenochaetus/
│   │   ├── Naso/
│   │   ├── Paracanthurus/
│   │   ├── Prionurus/
│   │   └── Zebrasoma/
│   ├── genes/                         # Candidate-gene lists from the literature review
│   │   ├── genes_coloring.txt
│   │   ├── genes_stripes.txt
│   │   └── genes_spots.txt
│   ├── genome_assemblies/             # NCBI assembly-availability search (metadata only)
│   │   ├── manifest.csv
│   │   └── README.md
│   └── phylogeny/                     # Fish Tree of Life genetic-data-only tree + coverage
│       ├── actinopt_12k_treePL.tre
│       ├── species_coverage.csv
│       ├── acanthuridae_synonyms.json
│       └── README.md
├── reports/
│   ├── image_sourcing_log.csv         # Per-image source URL, license, attribution
│   └── review.html                    # Human visual-review page (keep/reject)
├── src/
│   ├── dataset_builder/               # GBIF sourcing pipeline (see its docstring)
│   ├── fish_extractor/                # Fish identification & extraction (code-complete)
│   └── pattern_extractor/             # Coloring/stripe/spot feature extraction (prototype)
├── tests/
│   ├── dataset_builder/
│   ├── fish_extractor/
│   └── pattern_extractor/
├── BACKGROUND.md                      # Candidate-gene / pattern-genetics literature review
├── LICENSE
└── README.md
```

Everything else (distance matrices, phylogenetic analysis, and the scripts that produce them) will be added back stage by stage.

---

## Acknowledgements

This project is conducted in the **Dornburg Lab** at the University of North Carolina Charlotte. Phylogenetic reference data are obtained from the Fish Tree of Life (fishtreeoflife.org). GPU computation (fish detection/segmentation) runs on Google Colab; the Phase 2 pattern-extraction methodology is a Python reimplementation of *patternize* (Van Belleghem et al. 2018) — see Planned Approach for the full citation.
