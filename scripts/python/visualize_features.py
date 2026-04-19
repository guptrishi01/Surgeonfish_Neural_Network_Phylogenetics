"""
visualise_features.py

Generates a visual inspection report for the extracted features.
For each species, produces a multi-panel figure showing:

  Panel 1 -- Original standardised image with mask overlay (green)
  Panel 2 -- Masked fish pixels only (background zeroed)
  Panel 3 -- Hue histogram with colour-coded bars
  Panel 4 -- Saturation vs Value scatter of fish pixels (subsample)
  Panel 5 -- Gabor response heatmap (strongest orientation)
             FIX: vmax = np.percentile(fish_vals, 99) not global max
  Panel 6 -- LBP texture map on fish region
             FIX: vmax = np.percentile(fish_vals, 99) not global max
  Panel 7 -- Dominant colour swatches (k-means clusters)
  Panel 8 -- Dorsal/ventral split with mean colour bands

All panels are saved as individual PNGs and combined into a single
HTML report at outputs/visualisation/report.html for browser viewing.

Usage:
  # Generate figures for all species
  python scripts/python/visualise_features.py

  # Single species (faster, for spot-checking)
  python scripts/python/visualise_features.py --species "Acanthurus lineatus"

  # Only generate the HTML index, skip re-rendering figures
  python scripts/python/visualise_features.py --html-only

  # Limit to N species (useful for a quick sanity check)
  python scripts/python/visualise_features.py --limit 10
"""

import csv
import json
import sys
import logging
import argparse
import warnings
import colorsys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.stats import entropy as scipy_entropy
from skimage.feature import local_binary_pattern
from skimage.filters import gabor

# ---------------------------------------------------------------------------
# Paths
# FIX: script lives at fishy/scripts/python/, so three .parent calls reach
#      the fishy/ project root. Two .parent calls only reached fishy/scripts/.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # fishy/
DATA_DIR     = PROJECT_ROOT / "data"
STD_DIR      = DATA_DIR / "standardized_images"
FEAT_DIR     = DATA_DIR / "features"
OUT_DIR      = PROJECT_ROOT / "outputs"
VIS_DIR      = OUT_DIR / "visualisation"

FEAT_JSON    = FEAT_DIR / "features.json"

MASK_SEARCH_DIRS = [
    OUT_DIR / "all_predictions",
    OUT_DIR / "test_predictions",
    OUT_DIR / "val_predictions",
    OUT_DIR / "predictions",
]

# Gabor parameters (must match extract_features.py)
GABOR_THETAS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GABOR_FREQS  = [0.1, 0.2, 0.4]
N_COLORS     = 5
LBP_RADIUS   = 1
LBP_POINTS   = 8

# Colour scheme
BG       = "#0f1117"
PANEL_BG = "#1a1d27"
TEXT_COL = "#e8e8f0"
SUBTLE   = "#4a4d5e"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_mask(genus: str, species_name: str) -> Path:
    """
    Locate the binary mask file for a species.

    FIX: mask files are named "{Genus} {species}_mask.png" (space-separated,
    _mask suffix). Previous version used "{species_stem}_mask.png" without
    the genus prefix, causing all 63 species to be skipped.
    """
    stem = f"{species_name}_mask"
    for d in MASK_SEARCH_DIRS:
        p = d / f"{stem}.png"
        if p.exists():
            return p
    return None


def load_mask_robust(path: Path, W: int, H: int) -> np.ndarray:
    """Return boolean mask [H, W], handling L and RGB formats."""
    img = Image.open(path)
    if img.mode == "RGB":
        arr = np.array(img, dtype=np.uint8)
        if np.allclose(arr[:, :, 0], arr[:, :, 1]) and \
           np.allclose(arr[:, :, 0], arr[:, :, 2]):
            mask_np = arr[:, :, 0]
        else:
            mask_np = np.array(img.convert("L"), dtype=np.uint8)
    elif img.mode == "RGBA":
        mask_np = np.array(img.convert("L"), dtype=np.uint8)
    else:
        mask_np = np.array(img, dtype=np.uint8)

    if img.size != (W, H):
        pil_resized = Image.fromarray(mask_np).resize((W, H), Image.NEAREST)
        mask_np = np.array(pil_resized, dtype=np.uint8)

    return mask_np > 127


# ---------------------------------------------------------------------------
# Per-species figure
# ---------------------------------------------------------------------------

def make_species_figure(
    species_name: str,
    genus: str,
    img_path: Path,
    mask_path: Path,
    feat_vector: np.ndarray,
) -> Path:
    """
    Generate the 8-panel inspection figure for one species.
    Returns the path to the saved PNG.
    """
    img_np  = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    H, W    = img_np.shape[:2]
    mask    = load_mask_robust(mask_path, W, H)

    if mask.sum() < 500:
        logger.warning(f"  Skipping {species_name} -- mask too small ({mask.sum()} px)")
        return None

    # Pre-compute everything needed for panels
    hsv         = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    masked_hsv  = hsv[mask]

    # Masked image (background = dark grey)
    masked_img        = img_np.copy()
    masked_img[~mask] = [30, 30, 30]

    # Hue histogram
    h_hist, h_edges = np.histogram(masked_hsv[:, 0], bins=18, range=(0, 180), density=True)

    # Saturation / Value scatter (subsample 3000 pixels)
    rng  = np.random.RandomState(42)
    n_px = mask.sum()
    if n_px > 3000:
        idx  = rng.choice(n_px, 3000, replace=False)
        sv_s = masked_hsv[idx, 1].astype(float) / 255.0
        sv_v = masked_hsv[idx, 2].astype(float) / 255.0
        sv_h = masked_hsv[idx, 0].astype(float) / 180.0
    else:
        sv_s = masked_hsv[:, 1].astype(float) / 255.0
        sv_v = masked_hsv[:, 2].astype(float) / 255.0
        sv_h = masked_hsv[:, 0].astype(float) / 180.0

    scatter_colours = np.array([
        colorsys.hsv_to_rgb(h, max(s, 0.3), max(v, 0.5))
        for h, s, v in zip(sv_h, sv_s, sv_v)
    ])

    # Gabor -- pick the orientation/freq with highest mean response on fish pixels
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    best_energy     = None
    best_label      = ""
    best_energy_val = -1.0
    for theta in GABOR_THETAS:
        for freq in GABOR_FREQS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                real, imag = gabor(gray, frequency=freq, theta=theta)
            energy = np.sqrt(real ** 2 + imag ** 2)
            val    = float(energy[mask].mean())
            if val > best_energy_val:
                best_energy_val = val
                best_energy     = energy
                best_label      = f"theta={int(np.degrees(theta))}deg  freq={freq}"

    # LBP map
    gray_uint8 = (gray * 255).astype(np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lbp = local_binary_pattern(gray_uint8, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    lbp_display        = lbp.copy()
    lbp_display[~mask] = 0

    # Dominant colour swatches from feature vector
    dc_start    = 34
    dom_centers = feat_vector[dc_start:dc_start + 15].reshape(5, 3)  # [5, HSV norm]
    dom_freqs   = feat_vector[dc_start + 15:dc_start + 20]

    # Dorsal/ventral split
    ys, _     = np.where(mask)
    y_mid     = (int(ys.min()) + int(ys.max())) / 2
    dorsal_m  = mask.copy(); dorsal_m[int(y_mid):, :]  = False
    ventral_m = mask.copy(); ventral_m[:int(y_mid), :] = False

    def mean_rgb(region_mask):
        px = img_np[region_mask]
        return px.mean(axis=0).astype(np.uint8) if len(px) > 0 else np.array([128, 128, 128])

    dors_rgb = mean_rgb(dorsal_m)
    vent_rgb = mean_rgb(ventral_m)

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.suptitle(
        f"{genus} {species_name}",
        color=TEXT_COL, fontsize=16, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.38, wspace=0.32,
        left=0.04, right=0.97, top=0.93, bottom=0.05
    )

    def style(ax, title):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(colors=SUBTLE, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(SUBTLE); sp.set_linewidth(0.5)
        ax.xaxis.label.set_color(SUBTLE)
        ax.yaxis.label.set_color(SUBTLE)

    # Panel 1: original + mask overlay
    ax1 = fig.add_subplot(gs[0, 0])
    style(ax1, "Image + mask overlay")
    overlay     = img_np.copy()
    green_layer = np.zeros_like(img_np)
    green_layer[mask] = [0, 200, 0]
    overlay_vis = cv2.addWeighted(overlay, 0.65, green_layer, 0.35, 0)
    ax1.imshow(overlay_vis)
    ax1.axis("off")
    ax1.text(10, H - 20, f"mask area: {mask.mean():.1%}  pixels: {mask.sum():,}",
             color=TEXT_COL, fontsize=7)

    # Panel 2: masked fish only
    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2, "Masked fish (background removed)")
    ax2.imshow(masked_img)
    ax2.axis("off")

    # Panel 3: hue histogram with colour bars
    ax3 = fig.add_subplot(gs[0, 2])
    style(ax3, "Hue histogram (fish pixels only)")
    bin_colours = [colorsys.hsv_to_rgb(((i * 10) + 5) / 360, 0.85, 0.85) for i in range(18)]
    ax3.bar(range(18), h_hist, color=bin_colours, edgecolor=BG, linewidth=0.4)
    ax3.set_xticks([0, 3, 6, 9, 12, 15, 18])
    ax3.set_xticklabels(
        ["0deg", "30deg", "60deg", "90deg", "120deg", "150deg", "180deg"],
        color=SUBTLE, fontsize=7
    )
    ax3.set_ylabel("Density", color=SUBTLE, fontsize=8)
    entropy_val = float(scipy_entropy(h_hist / (h_hist.sum() + 1e-8) + 1e-8))
    ax3.set_title(f"Hue histogram  (entropy={entropy_val:.2f})",
                  color=TEXT_COL, fontsize=10, fontweight="bold", pad=6)

    # Panel 4: S vs V scatter coloured by actual hue
    ax4 = fig.add_subplot(gs[0, 3])
    style(ax4, "Saturation vs Brightness (fish pixels)")
    ax4.scatter(sv_s, sv_v, c=scatter_colours, s=3, alpha=0.4, linewidths=0)
    ax4.set_xlabel("Saturation", fontsize=8)
    ax4.set_ylabel("Value (brightness)", fontsize=8)
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)

    # Panel 5: Gabor response heatmap
    # FIX: vmax is the 99th percentile of fish-pixel energy values, not the
    # global array maximum. Prevents smooth-bodied fish from rendering as
    # near-black when their absolute texture energy is low.
    ax5 = fig.add_subplot(gs[1, 0])
    style(ax5, f"Gabor response  ({best_label})")
    gabor_display        = best_energy.copy()
    gabor_display[~mask] = 0
    fish_vals = best_energy[mask]
    vmax_val  = float(np.percentile(fish_vals, 99)) if len(fish_vals) > 0 else None
    im = ax5.imshow(gabor_display, cmap="inferno", vmin=0, vmax=vmax_val)
    ax5.axis("off")
    plt.colorbar(im, ax=ax5, fraction=0.03, pad=0.02).ax.tick_params(
        colors=SUBTLE, labelsize=7
    )

    # Panel 6: LBP texture map
    # FIX: same percentile normalisation applied to LBP values.
    ax6 = fig.add_subplot(gs[1, 1])
    style(ax6, "LBP texture map (fish region)")
    lbp_fish_vals = lbp[mask]
    lbp_vmax      = float(np.percentile(lbp_fish_vals, 99)) if len(lbp_fish_vals) > 0 else None
    ax6.imshow(lbp_display, cmap="plasma", vmin=0, vmax=lbp_vmax)
    ax6.axis("off")

    # Panel 7: dominant colour swatches
    ax7 = fig.add_subplot(gs[1, 2])
    style(ax7, "Dominant colour clusters (k-means)")
    ax7.set_xlim(0, 1); ax7.set_ylim(0, 1); ax7.axis("off")

    for k in range(N_COLORS):
        h_n, s_n, v_n = dom_centers[k]
        freq           = dom_freqs[k]
        rgb            = colorsys.hsv_to_rgb(h_n, s_n, v_n)
        swatch_w       = freq
        x_pos          = sum(dom_freqs[:k])
        rect = Rectangle((x_pos, 0.2), swatch_w, 0.6,
                          facecolor=rgb, edgecolor=BG, linewidth=1)
        ax7.add_patch(rect)
        if freq > 0.08:
            ax7.text(x_pos + swatch_w / 2, 0.5,
                     f"{freq:.0%}", ha="center", va="center",
                     color="white" if v_n < 0.6 else "black",
                     fontsize=9, fontweight="bold")
    ax7.text(0.5, 0.1, "Width = proportion of fish covered",
             ha="center", color=SUBTLE, fontsize=7)

    # Panel 8: dorsal/ventral split
    ax8 = fig.add_subplot(gs[1, 3])
    style(ax8, "Dorsal / ventral mean colour")
    split_vis        = img_np.copy()
    split_vis[~mask] = [30, 30, 30]
    ax8.imshow(split_vis)
    ax8.axhline(y=y_mid, color="cyan", lw=2, ls="--", alpha=0.8)

    d_rgb_norm = dors_rgb / 255.0
    v_rgb_norm = vent_rgb / 255.0

    ax8.text(W * 0.05, y_mid * 0.4,
             f"Dorsal\nR={dors_rgb[0]} G={dors_rgb[1]} B={dors_rgb[2]}",
             color=TEXT_COL, fontsize=7,
             bbox=dict(facecolor=tuple(d_rgb_norm) + (0.7,), edgecolor="none",
                       boxstyle="round"))
    ax8.text(W * 0.05, y_mid * 1.5,
             f"Ventral\nR={vent_rgb[0]} G={vent_rgb[1]} B={vent_rgb[2]}",
             color=TEXT_COL, fontsize=7,
             bbox=dict(facecolor=tuple(v_rgb_norm) + (0.7,), edgecolor="none",
                       boxstyle="round"))
    ax8.axis("off")

    # Save
    safe_stem = species_name.replace(" ", "_").replace("/", "_")
    out_path  = VIS_DIR / "figures" / f"{genus}_{safe_stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# HTML report builder
# ---------------------------------------------------------------------------

def build_html_report(figure_paths: list) -> Path:
    """
    Build a single HTML file that references all species figures via relative
    paths for browser-based visual inspection.

    figure_paths is a list of (fig_path, species_name, genus) tuples.

    To view the report, copy the entire visualisation/ directory to a local
    machine (both report.html and the figures/ subdirectory must be present)
    and open report.html in any browser.
    """
    html_path = VIS_DIR / "report.html"

    rows = []
    for fig_path, species_name, genus in figure_paths:
        rel    = fig_path.relative_to(VIS_DIR)
        anchor = f'{genus}_{species_name.replace(" ", "_")}'
        rows.append(f"""
        <div class="card" id="{anchor}">
          <div class="label">{genus} <em>{species_name}</em></div>
          <img src="{rel}" alt="{species_name}" loading="lazy">
        </div>""")

    toc = "".join(
        f'<a href="#{g}_{s.replace(" ", "_")}">{g[:3]}. {s}</a>'
        for _, s, g in figure_paths
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Surgeonfish Feature Visualisation</title>
<style>
  body  {{ background:#0f1117; color:#e8e8f0; font-family:sans-serif;
           margin:0; padding:20px; }}
  h1    {{ font-size:1.4rem; font-weight:500; margin-bottom:4px; }}
  p     {{ color:#4a4d5e; font-size:0.85rem; margin-top:0; }}
  .grid {{ display:grid; grid-template-columns:1fr;
           gap:24px; max-width:1600px; margin:auto; }}
  .card {{ background:#1a1d27; border-radius:8px;
           padding:12px 16px; border:0.5px solid #2a2d3e; }}
  .label{{ font-size:1rem; font-weight:500; margin-bottom:8px; }}
  .label em {{ color:#4fc3f7; font-style:normal; }}
  img   {{ width:100%; border-radius:4px; display:block; }}
  .toc  {{ background:#1a1d27; border-radius:8px; padding:12px 16px;
           margin-bottom:24px; border:0.5px solid #2a2d3e;
           max-width:1600px; margin:0 auto 24px; }}
  .toc a{{ color:#4fc3f7; text-decoration:none; margin-right:16px;
           font-size:0.82rem; }}
  .toc a:hover {{ text-decoration:underline; }}
</style>
</head>
<body>
<h1>Surgeonfish Feature Visualisation Report</h1>
<p>{len(figure_paths)} species &nbsp;|&nbsp;
   Each panel: mask overlay, masked fish, hue histogram, S/V scatter,
   Gabor response (per-fish 99th-pct normalisation),
   LBP texture (per-fish 99th-pct normalisation),
   dominant colours, dorsal/ventral split</p>

<div class="toc">{toc}</div>

<div class="grid">
{"".join(rows)}
</div>
</body>
</html>"""

    html_path.write_text(html, encoding="utf-8")
    return html_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visual inspection of extracted fish features"
    )
    parser.add_argument("--species",   type=str,  default=None,
                        help="Species name to visualise (partial match, e.g. 'lineatus')")
    parser.add_argument("--limit",     type=int,  default=None,
                        help="Limit to first N species (for quick check)")
    parser.add_argument("--html-only", action="store_true",
                        help="Rebuild HTML report without re-rendering figures")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("FEATURE VISUALISATION")
    logger.info("=" * 70)
    logger.info(f"Project root : {PROJECT_ROOT}")
    logger.info(f"Features JSON: {FEAT_JSON}")
    logger.info(f"Output dir   : {VIS_DIR}")

    if not FEAT_JSON.exists():
        logger.error(f"features.json not found: {FEAT_JSON}")
        logger.error("Run extract_features.py first.")
        sys.exit(1)

    with open(FEAT_JSON) as f:
        feat_data = json.load(f)

    species_entries = feat_data["species"]

    if args.species:
        species_entries = [
            s for s in species_entries
            if args.species.lower() in s["species"].lower()
        ]
        if not species_entries:
            logger.error(f"No species matching '{args.species}' found.")
            sys.exit(1)

    if args.limit:
        species_entries = species_entries[:args.limit]

    logger.info(f"Generating figures for {len(species_entries)} species")

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    figure_records = []

    if not args.html_only:
        for entry in species_entries:
            species_name = entry["species"]
            genus        = entry["genus"]
            img_path     = Path(entry["image_path"])
            feat_vector  = np.array(entry["features"], dtype=np.float32)

            # FIX: pass genus to find_mask so the correct filename pattern
            # "{Genus} {species}_mask.png" is constructed.
            mask_path = find_mask(genus, species_name)
            if mask_path is None:
                logger.warning(f"  [SKIP] {genus} {species_name} -- no mask found")
                logger.warning(f"         Searched for: '{genus} {species_name}_mask.png'")
                logger.warning(f"         In dirs: {[str(d) for d in MASK_SEARCH_DIRS]}")
                continue

            logger.info(f"  Rendering: {genus} {species_name}")
            fig_path = make_species_figure(
                species_name, genus, img_path, mask_path, feat_vector
            )
            if fig_path is not None:
                figure_records.append((fig_path, species_name, genus))
    else:
        # --html-only: collect existing figures from disk without re-rendering
        for entry in species_entries:
            species_name = entry["species"]
            genus        = entry["genus"]
            safe_stem    = species_name.replace(" ", "_").replace("/", "_")
            fig_path     = VIS_DIR / "figures" / f"{genus}_{safe_stem}.png"
            if fig_path.exists():
                figure_records.append((fig_path, species_name, genus))
            else:
                logger.warning(f"  [MISSING] {fig_path.name}")

    if not figure_records:
        logger.error("No figures generated or found.")
        sys.exit(1)

    html_path = build_html_report(figure_records)
    logger.info(f"\nHTML report : {html_path}")
    logger.info(f"Figures dir : {VIS_DIR / 'figures'}/")
    logger.info(f"Species rendered: {len(figure_records)}")
    logger.info("")
    logger.info("To view: copy outputs/visualisation/ to your local machine")
    logger.info("  scp -r rgupta25@gal-i9:~/fishy/outputs/visualisation/ .")
    logger.info("  Then open visualisation/report.html in a browser.")


if __name__ == "__main__":
    main()