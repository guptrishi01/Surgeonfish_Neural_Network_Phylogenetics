"""Human visual review: a static HTML page plus a feedback-driven re-run.

Automated filters (resolution, duplicates, aspect ratio, border busyness)
catch mechanical problems, but "is this fish actually close-up with a
clean background" turned out not to be reliably decidable from generic
image statistics - a pilot run showed both a global edge-density check and
a center-crop sharpness check disagreeing with visual judgment, in one case
flagging the single best photo in the batch as bad. Rather than keep
chasing a heuristic, a person looks at the grid and says keep/reject; that
decision flows back into the pipeline's own state so rejected candidates
are never re-downloaded and the vacated slots get backfilled.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from dataset_builder.config import PipelineConfig
from dataset_builder.state import PipelineState

logger = logging.getLogger(__name__)

_OCCURRENCE_KEY_PATTERN = re.compile(r"_gbif_(\d+)\.")


@dataclass
class ImageEntry:
    """One image to render on the review page.

    Attributes:
        filename: Name of the file within its species folder.
        relative_src: Path to the image, relative to the review HTML file.
        is_reference: True for the original seed photo (not rejectable).
        license_url: Empty for the reference image.
        rights_holder: Empty for the reference image.
    """

    filename: str
    relative_src: str
    is_reference: bool
    license_url: str
    rights_holder: str


def _load_metadata(metadata_csv_path: Path) -> dict[tuple[str, str], dict]:
    """Maps (species, filename) -> its sourcing metadata row, if logged."""
    if not metadata_csv_path.exists():
        return {}
    with open(metadata_csv_path, newline="", encoding="utf-8") as f:
        return {(row["species"], row["filename"]): row for row in csv.DictReader(f)}


def generate_review_html(config: PipelineConfig, output_path: Path) -> Path:
    """Builds a self-contained HTML page for visually reviewing every species.

    Images are referenced by relative file path (not embedded), so the
    page must stay in a fixed position relative to raw_images_root - by
    default both live under the repo root, which keeps the relative path
    stable.

    Args:
        config: Pipeline configuration (used for raw_images_root and the
            metadata CSV location).
        output_path: Where to write the HTML file.

    Returns:
        The path written to.
    """
    metadata = _load_metadata(config.metadata_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    species_sections = []
    for genus_dir in sorted(p for p in config.raw_images_root.iterdir() if p.is_dir()):
        for species_dir in sorted(p for p in genus_dir.iterdir() if p.is_dir()):
            species_name = species_dir.name.replace("_", " ")
            entries = []
            for image_path in sorted(species_dir.glob("*")):
                is_reference = image_path.stem == "000_reference"
                row = metadata.get((species_name, image_path.name), {})
                entries.append(
                    ImageEntry(
                        filename=image_path.name,
                        relative_src=str(Path("..") / image_path).replace("\\", "/"),
                        is_reference=is_reference,
                        license_url=row.get("license_url", ""),
                        rights_holder=row.get("rights_holder", ""),
                    )
                )
            species_sections.append((species_name, entries))

    html = _render_html(species_sections, target=config.target_per_species)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Wrote review page (%d species) to %s", len(species_sections), output_path)
    return output_path


def _render_html(species_sections: list[tuple[str, list[ImageEntry]]], target: int) -> str:
    cards = []
    for species_name, entries in species_sections:
        count = len(entries)
        status_class = "ok" if count >= target else "short"
        image_cards = []
        for entry in entries:
            if entry.is_reference:
                image_cards.append(f"""
                <div class="image-card reference">
                  <img src="{entry.relative_src}" loading="lazy">
                  <div class="caption">reference (pre-existing)</div>
                </div>""")
            else:
                image_cards.append(f"""
                <div class="image-card">
                  <img src="{entry.relative_src}" loading="lazy">
                  <label>
                    <input type="checkbox" class="keep-box" checked
                           data-species="{species_name}" data-filename="{entry.filename}">
                    keep
                  </label>
                  <div class="caption">{entry.rights_holder}</div>
                </div>""")
        cards.append(f"""
        <section class="species {status_class}">
          <h2>{species_name} <span class="count">{count}/{target}</span></h2>
          <div class="grid">{"".join(image_cards)}</div>
        </section>""")

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Surgeonfish image review</title>
<style>
  body {{
    font-family: system-ui, sans-serif; margin: 0; padding: 1.5rem;
    background: #111; color: #eee;
  }}
  h1 {{ font-size: 1.3rem; }}
  h2 {{ font-size: 1rem; display: flex; gap: 0.5rem; align-items: baseline; }}
  .count {{ font-weight: normal; opacity: 0.7; font-size: 0.85rem; }}
  section.short h2 .count {{ color: #ff8080; }}
  section.ok h2 .count {{ color: #80ff9c; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1.5rem; }}
  .image-card {{
    width: 180px; background: #1c1c1c; border-radius: 6px; padding: 0.4rem;
  }}
  .image-card.reference {{ outline: 2px solid #6ab0ff; }}
  .image-card img {{
    width: 100%; height: 180px; object-fit: cover; border-radius: 4px;
  }}
  .caption {{
    font-size: 0.7rem; opacity: 0.6; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
  }}
  label {{ font-size: 0.8rem; display: flex; gap: 0.3rem; align-items: center; }}
  #toolbar {{
    position: sticky; top: 0; background: #111; padding: 0.75rem 0; z-index: 10;
  }}
  button {{ font-size: 0.9rem; padding: 0.5rem 1rem; cursor: pointer; }}
  #instructions {{ font-size: 0.8rem; opacity: 0.8; max-width: 60rem; }}
</style>
</head>
<body>
<div id="toolbar">
  <h1>Surgeonfish image review</h1>
  <p id="instructions">
    Uncheck any image that is not an acceptable close-up (fish too small/distant,
    background too busy, wrong species, etc).
    The reference photo (blue outline) cannot be rejected here. When done, click
    "Export rejections" - it downloads <code>review_feedback.json</code>. Move that file to
    <code>reports/review_feedback.json</code>, then re-run:
    <code>python -m dataset_builder.cli --apply-review reports/review_feedback.json</code>
    to delete the rejected images and backfill each species back to target.
  </p>
  <button id="export">Export rejections</button>
  <span id="export-status"></span>
</div>
{"".join(cards)}
<script>
document.getElementById("export").addEventListener("click", () => {{
  const rejected = {{}};
  document.querySelectorAll(".keep-box").forEach((box) => {{
    if (!box.checked) {{
      const species = box.dataset.species;
      const filename = box.dataset.filename;
      (rejected[species] ||= []).push(filename);
    }}
  }});
  const blob = new Blob([JSON.stringify(rejected, null, 2)], {{type: "application/json"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "review_feedback.json";
  a.click();
  URL.revokeObjectURL(url);
  const total = Object.values(rejected).reduce((n, arr) => n + arr.length, 0);
  const status = document.getElementById("export-status");
  status.textContent = ` ${{total}} image(s) marked for rejection.`;
}});
</script>
</body>
</html>
"""


def apply_review_feedback(config: PipelineConfig, feedback_path: Path) -> dict[str, int]:
    """Deletes rejected images and resets state so backfilling can resume.

    Args:
        config: Pipeline configuration.
        feedback_path: Path to a review_feedback.json produced by the
            review page's "Export rejections" button:
            ``{"Genus species": ["003_gbif_12345.jpg", ...], ...}``.

    Returns:
        Mapping of species name to how many images were removed.
    """
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    state = PipelineState(config.state_path)
    removed_counts: dict[str, int] = {}

    for species_name, filenames in feedback.items():
        species_state = state.get(species_name)
        genus = species_name.split(" ", 1)[0]
        species_dir = config.raw_images_root / genus / species_name.replace(" ", "_")
        removed = 0

        for filename in filenames:
            if filename.startswith("000_reference"):
                logger.warning("Refusing to remove reference image for %s", species_name)
                continue
            path = species_dir / filename
            if path.exists():
                path.unlink()
                removed += 1
            match = _OCCURRENCE_KEY_PATTERN.search(filename)
            if match:
                species_state.seen_occurrence_keys.add(int(match.group(1)))

        if removed:
            remaining = sorted(species_dir.glob("*"))
            species_state.accepted_count = len(remaining)
            species_state.accepted_hashes = _rehash(remaining)
            species_state.exhausted = False
            removed_counts[species_name] = removed
            logger.info(
                "%s: removed %d rejected image(s), %d remaining, will backfill",
                species_name, removed, species_state.accepted_count,
            )

    state.save()
    return removed_counts


def _rehash(image_paths: list[Path]) -> list[int]:
    from PIL import Image

    from dataset_builder.quality_filter import average_hash

    hashes = []
    for path in image_paths:
        try:
            image = Image.open(path)
            image.load()
        except Exception:  # noqa: BLE001
            continue
        hashes.append(average_hash(image))
    return hashes
