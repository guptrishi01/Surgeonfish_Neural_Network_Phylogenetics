# Phylogeny source — genetic-data coverage

Resolves the blocking finding from the external README audit (2026-08-20): the
Fish Tree of Life's "complete" phylogeny places species without genetic data via
*stochastic polytomy resolution* (birth-death process conditioned on taxonomy,
not phylogenetic information), and its own documentation warns these placements
"should generally not be used for downstream analyses of trait evolution"
(Rabosky 2015, *No substitute for real data*, cited by the `fishtree` R package).
A Mantel/Kmult test of pattern distance against phylogenetic distance is exactly
that kind of analysis, so using the complete tree unmodified would let 15 species'
worth of taxonomically-imputed (not genetically-informed) distances enter the test
statistic as if they carried real signal.

## What this is

`actinopt_12k_treePL.tre` — the Fish Tree of Life's **genetic-data-only**
chronogram (sampled taxa, not the polytomy-resolved "complete" tree), downloaded
directly from `https://fishtreeoflife.org/downloads/actinopt_12k_treePL.tre.xz`
on 2026-08-20 (robots.txt permits automated access; no `Disallow` rules at all).
11,638 tips total, matching the count reported in Rabosky et al. (2018).

`species_coverage.csv` — for each of the 64 study species, whether it appears as
a tip in that genetic-data-only tree (`has_genetic_data`).

## Result

**49 of 64 species (76.6%) have real genetic-data placement.** Well above the
Fish Tree of Life's global average (~37%, 11,638/31,526) - expected, since
Acanthuridae is a well-studied reef-fish family. The 15 without genetic data:

Acanthurus albimento, A. albipectoralis, A. chronixis, A. fowleri, A. gahhm,
A. grammoptilus, A. maculiceps, A. sohal, A. tractus, A. tristis,
Ctenochaetus marginatus, Naso tergus, Prionurus chrysurus, Zebrasoma veliferum,
Z. xanthurum.

## Decision

Phase 4 restricts the phylogenetic-distance side of the analysis to the 49
genetically-sampled species, using this tree directly (no polytomy resolution,
no imputed placements) rather than the complete/resolved tree. The 15 dropped
species are documented here, not silently excluded - this narrows the
comparative sample but means every patristic distance used is real phylogenetic
signal, not a taxonomy-conditioned random draw.

## Alternative considered, not used

Sorenson et al. (2013), *A multi-locus timetree of surgeonfishes (Acanthuridae,
Percomorpha), with revised family taxonomy*, Molecular Phylogenetics and
Evolution 68(1), 150-160, doi:10.1016/j.ympev.2013.03.014 - a family-specific
timetree from nine genes, covering 76% of extant Acanthuridae diversity. Not
used as the primary source here since the Fish Tree of Life's genetic-data-only
tree already covers more of the 64 study species with a consistent, broadly-used
reference; worth revisiting if per-species overlap with Sorenson et al.'s taxon
set turns out better for any species currently dropped.

Sorenson et al. also "strongly support a paraphyletic *Acanthurus* and
*Ctenochaetus*" and recommend dissolving *Ctenochaetus* into *Acanthurus*. This
project keeps the two as separate genus labels (matching the Fish Tree of Life's
taxonomy and standard reference databases) since the analysis operates at the
species-tip level - the genus grouping is descriptive, not part of the
statistical test - but the paraphyly itself is the specific, checkable
prediction already named in the README's Species section.
