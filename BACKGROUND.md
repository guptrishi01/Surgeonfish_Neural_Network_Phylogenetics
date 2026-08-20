# Background: genetic architecture of fish colour pattern

*Supporting section for the Acanthuridae visual-similarity / phylogenetic-distance study.
Every factual claim below is drawn from one of the five references listed at the end;
gene assignments come from the accompanying extraction (56 NCBI-validated genes).*

## 1. Why pattern is decomposed into three dimensions

The central methodological commitment of this study is that "visual similarity" is not
one quantity. Coloring, stripes, and spots are treated as three separate species x species
distance matrices, each tested independently against phylogenetic distance. The
justification is developmental: these pattern classes are produced by different cellular
and genetic mechanisms, so a single blended similarity score would average across
processes that need not share an evolutionary history.

Fish are an unusually favourable system for this argument. Whereas mammals possess a
single chromatophore class (melanocytes), fish have six distinct chromatophore types
(Luo et al. 2021), and medaka alone carries leucophores, melanophores, xanthophores and
iridophores. Colour pattern in fish is therefore not a single pigment gradient but the
spatial arrangement of several independently specified cell populations — the
precondition for coloring, striping and spotting to be genetically separable at all.

## 2. Stripes: self-organization and a repeatedly evolving channel gene

Stripe formation in zebrafish (*Danio rerio*) is, as Podobnik et al. (2020) describe it,
a self-organizing process based on cell-contact mediated interactions between three
chromatophore types, with iridophores playing the leading role. Pattern is thus an
emergent property of interactions among cells, not a pre-drawn template — which is why
mutations in genes mediating those interactions reorganize the pattern globally rather
than deleting a local feature.

The strongest single result for the stripes dimension concerns *kcnj13* (the *obelix* /
*jaguar* locus), encoding an inwardly rectifying potassium channel. In *D. rerio*,
mutations in *kcnj13* (*obelix*) result in fewer and wider stripes. Critically for a
phylogenetic study, Podobnik et al. used reciprocal hemizygosity tests to identify
*obelix*/*Kcnj13* as the gene that evolved between *D. rerio* and its vertically barred
sibling species *D. aesculapii*, and complementation tests implicated divergence in
*Kcnj13* function in two further *Danio* species. Their conclusion is that the results
point towards repeated and independent evolution of this gene during colour pattern
diversification.

This matters for interpreting a Mantel result in either direction. A single gene whose
function diverges repeatedly and independently across a genus is a mechanism by which
similar striping can arise in non-sister lineages — i.e. a concrete developmental route
to convergence that would *weaken* a pattern-phylogeny correlation without implying the
pattern is non-genetic.

Genes assigned to the stripes dimension: *agrp2*, *aqp3a*, *asip1*, *ednrba*, *gja5b*, *kcnj13*, *mpv17*.

## 3. Spots: connexins and cell adhesion

Spotted patterns in *D. rerio* arise from a partly distinct gene set. Podobnik et al.
report that mutations in the connexin genes *Cx39.4* (*luchs*, NCBI *gja4*) and *Cx41.8*
(*leopard*, NCBI *gja5b*), and in the cell-adhesion gene *Igsf11* (*seurat*), lead to
spotted patterns — in contrast to *Kcnj13*, whose mutation instead alters stripe number
and width. The gap-junction and adhesion machinery governing heterotopic chromatophore
contact therefore biases the system toward spots, while channel function biases it toward
stripe geometry.

Independent support comes from a marine species. Lin et al. (2021) compared skin
transcriptomes of black-spotted skin, non-spotted skin and caudal fin in spotted scat
(*Scatophagus argus*), finding 1358, 2086 and 487 differentially expressed genes between
the three pairwise comparisons, and 134 common significantly differentially expressed
genes. Their named pigmentation candidates were *tyrp1*, *mitf*, *pmel*, *slc7a2*,
*tjp1*, *hsp70* and *mart-1*, with DEGs enriched in tyrosine metabolism, melanogenesis,
Wnt signalling and MAPK signalling. This is the closest available ecological analogue to
surgeonfish — a marine, reef-associated species rather than a freshwater laboratory
model — and it converges on melanosome-structural and melanocyte-regulatory genes rather
than the connexin route.

Genes assigned to the spots dimension: *gja4*, *gja5b*, *igsf11*, *mlana*, *tyrp1*.

## 4. Coloring: pigment cell specification and pigment synthesis

The coloring dimension draws on the broadest gene set (52 genes here), reflecting that
it aggregates two distinguishable processes: specification and migration of
neural-crest-derived pigment cells, and the biosynthetic pathways that fill them. Luo
et al. (2021) review both, covering genes involved in neural crest migration and
development alongside those involved in melanin-based coloration and other pigment types.

Three biosynthetic arms are represented in the extracted set:

- **Melanin synthesis and melanosome structure** — *tyr*, *tyrp1*, *dct*, *pmel*, *mlana*,
  *oca2*, *slc24a5*, *slc45a2*, *bace2*, *hps5*, *lrmda*
- **Pteridine synthesis** (yellow/orange, endogenous) — *gch2*, *spr*, *spra*, *sprb*, *xdh*
- **Carotenoid handling** (red/orange, dietary) — *scarb1*, *bco2*, *bco2a*. Luo et al.
  note that most species, fish included, cannot synthesise their own carotenoid pigments
  and must acquire them from the diet.

Upstream of all three sit the pigment-cell specification regulators — *mitf*/*mitfa*,
*sox10*, *pax3*, *pax7*, *foxd3* — and the receptor/ligand systems controlling
chromatophore number and survival: *kita* (*sparse*) and *kitlg*, *csf1ra* (*pfeffer*,
also called *fms*), *ednrba* (*rose*) and *edn3b*, and the melanocortin axis *mc1r*,
*pomca*, *asip1*, *asip2b*/*agrp2*.

The carotenoid arm carries an interpretive caveat specific to this study's inference. Because
carotenoid colour depends on diet, any red/orange component of surgeonfish coloration is
partly an ecological rather than a heritable signal. Two co-occurring reef species could
resemble each other in carotenoid-derived hue through shared diet alone, producing visual
similarity uncorrelated with ancestry. This is an ecological, not developmental, route to
the same convergence the study is designed to detect.

## 5. Statistical caveat on the Mantel approach

Harmon and Glor (2010) evaluated the Mantel test for exactly the two applications relevant
here — testing for phylogenetic signal, and testing for an evolutionary correlation
between two characters — and found it has poor performance compared to alternative
methods, including low power and, under some circumstances, inflated type-I error. They
identify phylogenetic permutations as a remedy for the inflated type-I error of three-way
Mantel tests, but note this test still has considerably lower power than independent
contrasts. Their recommendation is that use of the Mantel test should be restricted to
cases in which data can only be expressed as pairwise distances among taxa.

Two consequences for this study. First, that recommendation is arguably satisfied here:
pairwise visual similarity between species images is natively a distance, not a per-species
trait value, so a matrix-based test is the appropriate form. Second, the low-power finding
means a **null result must be reported as inconclusive rather than as evidence of absence** —
the test may simply lack power to detect a real association. With three dimensions tested
independently, the multiple-comparison correction further raises the detection threshold,
compounding the power problem. Stating this limitation pre-emptively is stronger than
having a reviewer raise it.

## 6. Why this gene set does not answer the study's question — and what it does do

None of the five references assays Acanthuridae. Every gene listed here is a candidate
identified in *Danio* species, spotted scat, medaka, or mammals. This gene set therefore
cannot support any claim about which loci generate surgeonfish patterns; asserting
otherwise would be unsupported extrapolation across roughly 200 million years of teleost
divergence.

What it does establish is the study's central design premise: that coloring, stripes and
spots are governed by mechanistically distinct pathways, and so warrant three separate
tests rather than one blended similarity score. The extraction supports this directly —
the stripe and spot gene sets recovered from the same organism and the same paper are
nearly disjoint, sharing only *gja5b*. Stripe geometry tracks channel function
(*kcnj13*), while spotting tracks gap-junction and adhesion genes (*gja4*, *gja5b*,
*igsf11*).

The set also supplies the vocabulary for interpreting the outcome. A significant
correlation on one dimension but not another is a biologically coherent result, not an
inconsistency, because the underlying pathways are separable.

## 7. Translational framing

Russo et al. (2022) provide the applied justification. They report that roughly 70% of
human genes have at least one zebrafish ortholog, and that over 80% of known human disease
genes — including oncogenes and tumour suppressors — are represented; in zebrafish skin,
pigment production derives from melanocytes belonging to the neural-crest-derived pigment
cell system, the same lineage as in humans. Pigment-pattern genetics in fish thus feeds
directly into melanoma and skin-disease modelling, which is the strongest available
argument that comparative pattern work has downstream biomedical relevance rather than
being purely descriptive.

---

## References

1. Podobnik, M., Frohnhöfer, H. G., Dooley, C. M., Eskova, A., Nüsslein-Volhard, C., &
   Irion, U. (2020). Evolution of the potassium channel gene *Kcnj13* underlies colour
   pattern diversification in *Danio* fish. *Nature Communications*, 11, 6230.
   https://doi.org/10.1038/s41467-020-20021-6

2. Lin, X., Tian, C., Huang, Y., Shi, H., & Li, G. (2021). Comparative transcriptome
   analysis identifies candidate genes related to black-spotted pattern formation in
   spotted scat (*Scatophagus argus*). *Animals*, 11(3), 765.
   https://doi.org/10.3390/ani11030765

3. Luo, M., Lu, G., Yin, H., Wang, L., Atuganile, M., & Dong, Z. (2021). Fish pigmentation
   and coloration: Molecular mechanisms and aquaculture perspectives. *Reviews in
   Aquaculture*, 13(4), 2395–2412. https://doi.org/10.1111/raq.12583

4. Harmon, L. J., & Glor, R. E. (2010). Poor statistical performance of the Mantel test in
   phylogenetic comparative analyses. *Evolution*, 64(7), 2173–2178.
   https://doi.org/10.1111/j.1558-5646.2010.00973.x

5. Russo, I., Sartor, E., Fagotto, L., Colombo, A., Tiso, N., & Alaibac, M. (2022). The
   Zebrafish model in dermatology: an update for clinicians. *Discover Oncology*, 13, 48.
   https://doi.org/10.1007/s12672-022-00511-3

### Database resource

NCBI Gene (National Center for Biotechnology Information, U.S. National Library of
Medicine). Used to validate every gene symbol and resolve mutant-locus synonyms and
species-specific paralogues. https://www.ncbi.nlm.nih.gov/gene

### Note on sources

All five references were parsed in full (71 pages total: Podobnik main text and
supplementary, Lin full text via PubMed Central, Luo review, Harmon & Glor, Russo). A
sixth file supplied as the Lin et al. supplementary material was found to be a different
study (mouse inhibin/pituitary work) and was excluded; the spotted scat 134-gene common
DEG list is therefore not incorporated, and the spots dimension reflects only the
candidates named in the article text.

### Citation verification

Volume, issue, article and page numbers in the list above were read directly from the
retrieved PDFs rather than recalled: `Nat Commun (2020) 11:6230`;
`Rev Aquac. 2021;13:2395-2412`; `Evolution 64-7: 2173-2178`;
`Discover Oncology (2022) 13:48`; `Animals 11(3):765` (PMC8001731, PMID 33802016).
Author lists were taken from the article front matter.
