# Figures

Regenerate with `python src/scripts/make_figures.py`. Every figure is built from
tracked result files or the extracted crops — no numbers are hard-coded in the
script, so a figure cannot drift from the data it claims to show.

| file | what it shows |
| --- | --- |
| `fig1_pipeline.png` | One image through all four stages: crop → SAM 2 mask → cutout → colour clusters |
| `fig2_results.png` | Headline result — both tests, all three pattern dimensions |
| `fig3_distance_matrices.png` | The four 49×49 distance matrices, species ordered by phylogeny |
| `fig4_validation.png` | Detector performance against 155 hand labels, and the stripe recalibration |
| `fig5_species_space.png` | Species in pattern feature space, coloured by genus |

`fig1_pipeline.png` needs `data/extracted_fish/` (not tracked — ~480MB); the
script skips that figure with a message rather than failing if it is absent.
