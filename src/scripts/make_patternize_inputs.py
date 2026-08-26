"""Follow-up #3: build a *confound-free* patternize comparison.

The first attempt compared `patternize::kImage()` against our clustering on the
same crop, but kImage clusters the whole raster - including the grey background
of the mask cutout - while pattern_extractor clusters masked-in pixels only. The
two were partitioning different pixel sets, so the 0.11-0.26 fraction
differences established nothing.

Fix: emit an image containing *only* the masked-in pixels, reshaped into a
rectangle. Then both implementations see an identical pixel multiset, no
background exists to be included or excluded, and any remaining difference is a
genuine algorithmic difference rather than an artifact of the comparison.

At most (width - 1) pixels are dropped to make the count rectangular - under
0.2% of a typical fish mask, and reported per image so it is not hidden.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2, vq

ROOT = Path(
    r"c:\Users\guptr\OneDrive\Documents\Surgeonfish_Neural_Network_Phylogenetics"
)
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "patternize_check"
OUT.mkdir(parents=True, exist_ok=True)
EXTRACTED = ROOT / "data" / "extracted_fish"

# Three species spanning the pattern range: strongly striped, plain, and a
# mid-case, so the comparison isn't made on one convenient image.
SAMPLES = [
    "Acanthurus/Acanthurus_lineatus/000_reference",
    "Zebrasoma/Zebrasoma_flavescens/000_reference",
    "Ctenochaetus/Ctenochaetus_striatus/000_reference",
]
K = 4
STRIP_WIDTH = 500
SEED = 0

summary = []
for key in SAMPLES:
    crop = EXTRACTED / f"{key}.png"
    mask_path = EXTRACTED / f"{key}_mask.png"
    if not crop.exists() or not mask_path.exists():
        print(f"SKIP (missing files): {key}")
        continue

    rgb = np.array(Image.open(crop).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    pixels = rgb[mask]                      # (N, 3) uint8, masked-in only

    n_rows = len(pixels) // STRIP_WIDTH
    kept = n_rows * STRIP_WIDTH
    dropped = len(pixels) - kept
    strip = pixels[:kept].reshape(n_rows, STRIP_WIDTH, 3)

    # Cluster with the project's own approach: fit centres on a subsample,
    # then assign every pixel. Centres are shared with R so both start
    # identically - kImage's startCenter argument takes the same matrix.
    rng = np.random.default_rng(SEED)
    sample = pixels[rng.choice(len(pixels), min(50000, len(pixels)), replace=False)]
    centres, _ = kmeans2(sample.astype(float), K, iter=100, minit="++", seed=SEED)
    labels, _ = vq(strip.reshape(-1, 3).astype(float), centres)
    fractions = np.sort(np.bincount(labels, minlength=K) / len(labels))[::-1]

    name = key.replace("/", "__")
    Image.fromarray(strip).save(OUT / f"{name}.png")
    np.savetxt(OUT / f"{name}_centres.csv", centres, delimiter=",")
    np.savetxt(OUT / f"{name}_fractions.csv", fractions, delimiter=",")

    summary.append({
        "image": name,
        "masked_pixels": int(len(pixels)),
        "pixels_compared": int(kept),
        "pixels_dropped": int(dropped),
        "dropped_pct": round(100 * dropped / len(pixels), 4),
        "python_fractions": " ".join(f"{v:.6f}" for v in fractions),
    })
    print(f"{name}")
    print(f"   masked pixels {len(pixels):>9,} | compared {kept:>9,} | "
          f"dropped {dropped} ({100*dropped/len(pixels):.3f}%)")
    print(f"   python fractions: {np.round(fractions, 4).tolist()}")

with open(OUT / "python_reference.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

(OUT / "README.md").write_text(
    "# patternize equivalence-check inputs\n\n"
    "Each `*.png` here holds **only the masked-in pixels** of one extracted fish\n"
    "crop, reshaped into a rectangle. There is no background, so\n"
    "`patternize::kImage()` and `pattern_extractor` cluster an identical pixel\n"
    "multiset - which the first version of this check did not achieve, making its\n"
    "result uninterpretable (see CHANGELOG v6.3.0).\n\n"
    "- `*_centres.csv` - the k-means starting centres, passed to `kImage()` as\n"
    "  `startCenter` so both implementations begin from the same point.\n"
    "- `*_fractions.csv` - our cluster fractions, sorted descending.\n"
    "- `python_reference.csv` - per-image pixel counts and fractions, including\n"
    "  how many pixels were dropped to make the count rectangular.\n\n"
    "Regenerate with `scratchpad/make_patternize_inputs.py`; compare against R via\n"
    "Part B of `notebooks/Followups.ipynb`.\n",
    encoding="utf-8",
)
print(f"\nwrote {len(summary)} image(s) to {OUT.relative_to(ROOT)}")
