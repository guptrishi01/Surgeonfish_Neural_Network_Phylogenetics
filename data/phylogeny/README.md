# Phylogeny source — genetic-data coverage

Resolves the blocking finding from the external README audit (2026-08-20): the
Fish Tree of Life's "complete" phylogeny places species without genetic data via
*stochastic polytomy resolution* (birth-death process conditioned on taxonomy,
not phylogenetic information), and its own documentation warns these placements
"should generally not be used for downstream analyses of trait evolution"
(Rabosky 2015, *No substitute for real data*, cited by the `fishtree` R package).
A Kmult/Mantel test of pattern distance against phylogenetic distance is exactly
that kind of analysis, so using the complete tree unmodified would let a chunk
of taxonomically-imputed (not genetically-informed) distances enter the test
statistic as if they carried real signal.

**Updated 2026-08-21** after a second external audit round caught a real bug in
the first version of this matching (see Result/Match method below): exact
string-matching missed one genuine synonym pair, undercounting coverage by one
species. Independently re-verified against the actual tree file before fixing -
see the note at the end on what the audit got right vs. overstated.

## What this is

`actinopt_12k_treePL.tre` — the Fish Tree of Life's **genetic-data-only**
chronogram (sampled taxa, not the polytomy-resolved "complete" tree), downloaded
directly from `https://fishtreeoflife.org/downloads/actinopt_12k_treePL.tre.xz`
on 2026-08-20 (robots.txt permits automated access; no `Disallow` rules at all).
11,638 tips total, matching the count reported in Rabosky et al. (2018).
SHA-256 of the decompressed file (for reproducibility - Fish Tree of Life
releases are versioned and this pins which one was used):
`95033b6f13cb07c474ef37dc7c03c0917dec4ed876b001ec8dd40063e569e5b2`.

`species_coverage.csv` — for each of the 64 study species: whether it appears
as a tip in that genetic-data-only tree (`has_genetic_data`), and how the match
was made (`match_method`: `exact`, `synonym (<tip name>)`, or blank if
unmatched).

`acanthuridae_synonyms.json` — NCBI Taxonomy synonym table for the family,
used to check unmatched species against the tree's 18 otherwise-unused
Acanthuridae tips via synonym resolution (not just exact string match).

## Result

**50 of 64 species (78.1%) have real genetic-data placement** - well above the
Fish Tree of Life's global average (~37%, 11,638/31,526), expected since
Acanthuridae is a well-studied reef-fish family. 49 matched by exact name;
1 more (*Zebrasoma veliferum*) matched via a confirmed synonym - the tree's tip
is labeled *Zebrasoma velifer*, the same taxon under NCBI's canonical name form
(same tax ID, already confirmed during the Phase 0 genome-assembly search).

The 14 still unmatched, by genus - **10 Acanthurus, 1 Ctenochaetus, 1 Naso,
1 Prionurus, 1 Zebrasoma**: A. albimento, A. albipectoralis, A. chronixis,
A. fowleri, A. gahhm, A. grammoptilus, A. maculiceps, A. sohal, A. tractus,
A. tristis, Ctenochaetus marginatus, Naso tergus, Prionurus chrysurus,
Zebrasoma xanthurum. This exclusion is concentrated in *Acanthurus* (already
the best-sampled genus at 31/52), not in the two genera flagged as
under-sampled in the README's Species section (*Naso*, *Prionurus* each lose
only 1) - so it doesn't meaningfully compound that existing gap.

**Important nuance, not "no genetic data exists":** spot-checked via NCBI
Nucleotide (`esearch`) - most of these 14 do have GenBank sequence records
(e.g. *A. tractus* 117, *A. albimento* 5, *Zebrasoma xanthurum* 27, *Naso
tergus* 2), while *A. chronixis* genuinely has zero. Absence from this tree
reflects which sequences were assembled into *this specific Fish Tree of Life
release's fitted supermatrix*, not absence of sequence data altogether.
Building a custom tree extension from raw GenBank sequence for these species
is out of scope for now, but is a legitimate future option if the sample size
becomes limiting.

## Decision

Phase 4 restricts the phylogenetic-distance side of the analysis to the 50
genetically-sampled species, using this tree directly (no polytomy resolution,
no imputed placements) rather than the complete/resolved tree. The 14 dropped
species are documented here, not silently excluded - this narrows the
comparative sample but means every patristic distance used is real phylogenetic
signal, not a taxonomy-conditioned random draw.

## What the second audit got right vs. overstated

Right: the 67-tip / 18-unused-tip counts were exact (independently
re-verified against the tree file), and one of the 15 originally-unmatched
species was a real, fixable synonym-matching bug (*Z. veliferum*/*velifer*).

Overstated: the framing implied most of the 18 unused tips represented
recoverable matches for the 15 dropped species. Checking the other 17 unused
tips directly, they're real, different Acanthuridae species genuinely not in
this study's 64 (e.g. *Acanthurus coeruleus*, *A. bahianus*, *Naso elegans*,
*Prionurus punctatus*) - not synonyms of anything we're missing. Running the
provided synonym table bidirectionally against the tree recovered exactly one
additional match, not more.

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
