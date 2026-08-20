"""Model 1 of the surgeonfish ML pipeline: fish identification & extraction.

Given a raw source photo, this package decides whether it shows exactly one,
roughly-centered fish and, if so, extracts it (mask cutout, cropped to the
mask's bounding box) for the next model (pattern/feature extraction, not yet
specified) to consume.

Architecture: Grounded SAM 2 - Grounding DINO does zero-shot, text-prompted
("fish.") open-vocabulary object detection, then SAM 2 turns the chosen box
into a precise segmentation mask. Both are used via Hugging Face
``transformers`` rather than their original repos, since Grounding DINO's
original IDEA-Research implementation requires compiling a custom CUDA op -
a common friction point on Colab, which is this project's execution target
(not the UNC Charlotte HPC/SLURM setup CLAUDE.md describes for other parts
of this pipeline).

No training happens in this phase: detection and segmentation both run
zero-shot. Every image that doesn't clearly show one centered fish is
flagged for human review rather than silently accepted or dropped, via the
same self-contained review-HTML pattern used in ``dataset_builder.review``.
"""
