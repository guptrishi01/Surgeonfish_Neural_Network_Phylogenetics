"""Model 2 of the surgeonfish ML pipeline: pattern extraction.

Operates on ``fish_extractor``'s output (mask-cutout crops, one clean fish
per image) and extracts three pattern dimensions per image: coloring,
stripes, spots/freckles - kept separate rather than blended into one
feature vector, per the developmental justification in BACKGROUND.md.

**Provenance.** The core color-region-identification step is a Python
reimplementation of the k-means method from *patternize*, an R package for
quantifying colour pattern variation (Van Belleghem, Papa, Planas, Martin,
Counterman, Papa, & Jiggins, 2018, "patternize: An R package for
quantifying colour pattern variation," *Methods in Ecology and Evolution*
9(2), 390-398, https://doi.org/10.1111/2041-210X.12853). patternize has
been demonstrated on fish and used for phylogenetic comparative colour
pattern studies, making it the closest published, peer-reviewed precedent
for this task.

Two deliberate departures from patternize's original method, both because
this study compares 64 different *species* rather than variation within one
species/population:

1. **No cross-specimen pixel registration.** patternize's landmark- or
   image-registration-based homology step assumes a broadly consistent body
   plan across compared specimens (e.g. butterfly wing polymorphism,
   guppy populations). Acanthuridae body shapes vary too much (*Naso*'s
   horns, *Zebrasoma*'s tall disc body) for that to be reliable, so
   clustering and pattern-region analysis run per-image rather than on a
   pixel-aligned stack.
2. **Stripe/spot geometry is this project's own extension**, built on top
   of patternize's clustering output (connected-component/region-shape
   analysis of each cluster's binary mask - see ``geometry.py``) rather
   than a capability patternize itself provides; patternize quantifies
   colour-pattern *regions* generally, not stripe vs. spot geometry
   specifically.

Reference-initialized k-means (``clustering.py``) is reimplemented with
``scipy.cluster.vq.kmeans2`` rather than adding scikit-learn as a new
dependency; connected-component/region analysis uses ``scipy.ndimage``
rather than scikit-image, for the same reason - this package needs no new
heavy dependencies beyond scipy, which is CPU-only and already required.
"""
