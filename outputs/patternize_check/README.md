# patternize equivalence-check inputs

Each `*.png` here holds **only the masked-in pixels** of one extracted fish
crop, reshaped into a rectangle. There is no background, so
`patternize::kImage()` and `pattern_extractor` cluster an identical pixel
multiset - which the first version of this check did not achieve, making its
result uninterpretable (see CHANGELOG v6.3.0).

- `*_centres.csv` - the k-means starting centres, passed to `kImage()` as
  `startCenter` so both implementations begin from the same point.
- `*_fractions.csv` - our cluster fractions, sorted descending.
- `python_reference.csv` - per-image pixel counts and fractions, including
  how many pixels were dropped to make the count rectangular.

Regenerate with `scratchpad/make_patternize_inputs.py`; compare against R via
Part B of `notebooks/Followups.ipynb`.
