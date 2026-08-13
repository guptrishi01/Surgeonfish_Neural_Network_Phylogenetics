# Surgeonfish Visual Phenomics and Phylogenetic Inference

## Version

**1.0.0** — see [CLAUDE.md](CLAUDE.md) for the versioning scheme (major = phase completion, minor = a step within a phase, patch = a bug fix).

**Changelog**
- **1.0.0** — Phase 0 (data collection) complete. ~1,850 candidate images sourced from GBIF across all 64 species, visually reviewed over 9 rounds (~490 images rejected and backfilled along the way). 55/64 species reached the 25-image target; the remaining 9 are genuinely limited by GBIF's available licensed photos, not a pipeline gap. Along the way: fixed a Windows encoding crash, a filename-numbering bug (with a cleanup pass over the existing dataset), and confirmed automated close-up/background filtering isn't reliable enough to replace human review.

This section will be trimmed down once the project is finished — for now it's tracking everything as it happens.

---

## Research Question

**Is visual pattern similarity among surgeonfish (family Acanthuridae) genetically associated with evolutionary relatedness — or does it arise independently of shared ancestry?**

Concretely: for a set of Acanthuridae species with a known molecular phylogeny, does quantified visual similarity (color pattern, texture) between species pairs correlate with their phylogenetic distance? A significant correlation (tested via Mantel test) is evidence that visual pattern is genetically associated with lineage — i.e. that closely related species tend to look alike because they share ancestry, not merely because they share habitat or ecology. A null result means visual pattern is not explained by relatedness at the level this method can detect, and any resemblance between species is more likely convergent.

This is a correlational test, not a claim of direct causation: the Mantel test establishes statistical association between visual-similarity and phylogenetic-distance matrices. It does not, on its own, identify the underlying genetic or developmental mechanism.

---

## Status

This project is being rebuilt from scratch after an earlier pipeline run produced unreliable results (a broken evaluation metric, a misreported ROC-AUC, a silently-dropped test image, and a phylogenetic-signal result that came from data-dredged feature selection rather than a preregistered test). Rebuilding proceeds one phase at a time (see [CLAUDE.md](CLAUDE.md)), each verified against its own output before moving to the next — see the changelog above for what's done so far.

Nothing in this README describes final scientific results yet — that section gets written last, once every number in it traces back to a real output file.

---

## Species

63 species across all six extant Acanthuridae genera, sourced from lateral reference photographs:

| Genus | Species count |
|---|---|
| *Acanthurus* | 31 |
| *Ctenochaetus* | 9 |
| *Naso* | 14 |
| *Paracanthurus* | 1 |
| *Prionurus* | 3 |
| *Zebrasoma* | 6 |
| **Total** | **64** |

A key biological complication: *Acanthurus* is paraphyletic with respect to *Ctenochaetus* in the molecular phylogeny — several *Ctenochaetus* species are nested inside the *Acanthurus* clade. Whether that paraphyly is also reflected in visual similarity is one specific, checkable prediction of the broader research question above.

The molecular phylogeny used as the evolutionary-relatedness reference comes from the Fish Tree of Life (fishtreeoflife.org), a time-calibrated Newick tree built from nine molecular loci.

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

1. **Segmentation** — a Mask R-CNN model (ResNet-50 FPN backbone) isolates each fish from its background, producing a pixel mask per image.
2. **Feature extraction** — a classical computer-vision feature vector (color histograms, dominant colors, dorsal-ventral color gradient, texture) is computed from within each mask only, excluding background.
3. **Distance matrices** — pairwise visual distance between species (from features) and pairwise patristic distance between species (from the molecular tree) are each assembled into a distance matrix.
4. **Statistical test** — a Mantel test compares the two matrices to determine whether visual similarity and phylogenetic relatedness are significantly correlated, with permutation-based significance testing.

Implementation details (exact feature set, model hyperparameters, split strategy, and how feature-group selection avoids circularity) are being finalized as each stage is rebuilt.

---

## Repository Structure

```
Surgeonfish_Neural_Network_Phylogenetics/
├── data/
│   └── raw_images/                    # One subfolder per species, one per genus
│       ├── Acanthurus/
│       │   └── Acanthurus_guttatus/
│       │       ├── 000_reference.jpg  # Pre-existing curated photo (seed image)
│       │       └── 001_gbif_....jpg   # GBIF-sourced, license-checked, reviewed
│       ├── Ctenochaetus/
│       ├── Naso/
│       ├── Paracanthurus/
│       ├── Prionurus/
│       └── Zebrasoma/
├── reports/
│   ├── image_sourcing_log.csv         # Per-image source URL, license, attribution
│   └── review.html                    # Human visual-review page (keep/reject)
├── src/
│   └── dataset_builder/               # GBIF sourcing pipeline (see its docstring)
├── tests/
│   └── dataset_builder/
├── LICENSE
└── README.md
```

Everything else (standardized images, annotations, trained model, extracted features, distance matrices, phylogenetic analysis, and the scripts that produce them) will be added back stage by stage.

---

## Acknowledgements

This project is conducted in the **Dornburg Lab** at the University of North Carolina Charlotte. Phylogenetic reference data are obtained from the Fish Tree of Life (fishtreeoflife.org). Training and computation are performed on the UNC Charlotte HPC cluster.
