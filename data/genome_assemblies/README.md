# Genome assembly availability — Acanthuridae

Supporting data for Phase 6 (deferred, contingent — see `todo.txt` / the rebuild plan).
This is a **metadata search only**: no genome sequence (FASTA) was downloaded, only the
NCBI Datasets catalog record for each assembly. Assembly files for the matched species
are multi-hundred-MB each; actually pulling and searching them is a separate, later step
(not yet done, not yet needed until Phase 4 results motivate it).

## What this answers

Whether whole-genome assemblies exist at all for the 64 species in this study — the
open question `BACKGROUND.md` (§6) and the rebuild plan's Phase 6 section left
unresolved. It does **not** answer whether the 56 candidate genes in `data/genes/`
are actually present/annotated in these assemblies — that requires either pulling
each assembly's gene annotation or querying NCBI Gene per organism, not done here.

## Method

Queried the NCBI Datasets API (`GET /genome/taxon/Acanthuridae/dataset_report`,
family-level, one call, no pagination needed — 24 total results) on 2026-08-20.
Cross-referenced each returned organism name against the 64 species folders under
`data/raw_images/` (one direct name-form correction confirmed via NCBI taxonomy
lookup: NCBI's canonical name is *Zebrasoma velifer*, which resolves to the same
taxon ID as *Zebrasoma veliferum*, the epithet form used in this dataset).

## Result

**16 of the 64 species (25%) have at least one public genome assembly**, several at
chromosome level (*Acanthurus chirurgus*, *Acanthurus tractus*). See `manifest.csv`
for the full list with accessions, assembly level, submitter, and BioProject.

Matched species: *Acanthurus chirurgus, A. grammoptilus, A. lineatus, A. nigricauda,
A. tennentii, A. tractus, Ctenochaetus hawaiiensis, C. striatus, Naso annulatus,
N. brachycentron, N. brevirostris, N. lituratus, N. minor, N. vlamingii, Zebrasoma
flavescens, Z. veliferum*.

5 additional Acanthuridae assemblies exist for species *not* in this study's 64
(*A. coeruleus, A. leucopareius, A. reversus, Naso thynnoides*) — included in
`manifest.csv` (`in_our_64_species=no`) since they could serve as close-relative
fallbacks if a matched species' assembly turns out low-quality, but not otherwise
used.

## Caveat

Per `BACKGROUND.md` §6: assembly *existence* for a species says nothing on its own
about whether the specific 56 candidate genes are present, intact, or expressed in a
color-pattern-relevant way in that species — that's a downstream annotation/ortholog
question, only worth pursuing if Phase 4 finds a significant result to follow up on.
