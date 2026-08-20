"""Human visual review for images the QA gate couldn't auto-accept.

Follows dataset_builder.review's pattern: a self-contained HTML page lists
every flagged image with its detected box (if any) and flag reason; a
person decides which are genuinely unusable and exports that as JSON,
consumed by apply_review_feedback() to mark them permanently excluded.
Anything left unresolved (not exported) stays flagged, so a later re-run
with adjusted QA-gate thresholds or model settings can revisit it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from fish_extractor.config import PipelineConfig
from fish_extractor.state import ExtractionState

logger = logging.getLogger(__name__)


@dataclass
class FlaggedEntry:
    """One flagged image to render on the review page.

    Attributes:
        image_key: Path to the source image, relative to raw_images_root.
        relative_src: Path to the image, relative to the review HTML file.
        reason: QA-gate reason code.
        box_style: Inline CSS for a percentage-positioned overlay div
            showing the detected box, or None if there was no box (e.g.
            reason is "no_detection").
    """

    image_key: str
    relative_src: str
    reason: str
    box_style: str | None


def _box_overlay_style(box: list[float] | None, image_size: tuple[int, int]) -> str | None:
    """Percentage-based inline CSS positioning the detected box over the image."""
    if box is None:
        return None
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1, _confidence = box
    left = x0 / width * 100
    top = y0 / height * 100
    box_width = (x1 - x0) / width * 100
    box_height = (y1 - y0) / height * 100
    return f"left:{left:.2f}%; top:{top:.2f}%; width:{box_width:.2f}%; height:{box_height:.2f}%;"


def generate_review_html(config: PipelineConfig, output_path: Path) -> Path:
    """Builds a self-contained HTML page for reviewing every flagged image.

    Args:
        config: Pipeline configuration (used for raw_images_root and the
            state file location).
        output_path: Where to write the HTML file.

    Returns:
        The path written to.
    """
    state = ExtractionState(config.state_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for image_key, entry in state.items():
        if entry.status != "flagged":
            continue
        image_path = config.raw_images_root / image_key
        box_style = None
        if entry.box is not None and image_path.exists():
            try:
                with Image.open(image_path) as image:
                    box_style = _box_overlay_style(entry.box, image.size)
            except Exception:  # noqa: BLE001
                logger.warning("Could not read %s for box overlay", image_path)
        entries.append(
            FlaggedEntry(
                image_key=image_key,
                relative_src=str(Path("..") / image_path).replace("\\", "/"),
                reason=entry.reason,
                box_style=box_style,
            )
        )

    html = _render_html(entries)
    output_path.write_text(html, encoding="utf-8")
    logger.info(
        "Wrote fish-extraction review page (%d flagged image(s)) to %s", len(entries), output_path
    )
    return output_path


def _render_html(entries: list[FlaggedEntry]) -> str:
    cards = []
    for entry in entries:
        overlay = (
            f'<div class="box-overlay" style="{entry.box_style}"></div>'
            if entry.box_style else ""
        )
        cards.append(f"""
        <div class="image-card">
          <div class="img-wrap">
            <img src="{entry.relative_src}" loading="lazy">
            {overlay}
          </div>
          <div class="reason">{entry.reason}</div>
          <label>
            <input type="checkbox" class="exclude-box" data-key="{entry.image_key}">
            exclude permanently
          </label>
        </div>""")

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Fish extraction review</title>
<style>
  body {{
    font-family: system-ui, sans-serif; margin: 0; padding: 1.5rem;
    background: #111; color: #eee;
  }}
  h1 {{ font-size: 1.3rem; }}
  #instructions {{ font-size: 0.8rem; opacity: 0.8; max-width: 60rem; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 0.75rem; }}
  .image-card {{ width: 220px; background: #1c1c1c; border-radius: 6px; padding: 0.4rem; }}
  .img-wrap {{ position: relative; }}
  .img-wrap img {{
    width: 100%; height: 200px; object-fit: contain; border-radius: 4px; background: #000;
  }}
  .box-overlay {{ position: absolute; border: 2px solid #ff5050; pointer-events: none; }}
  .reason {{ font-size: 0.75rem; opacity: 0.8; margin: 0.3rem 0; }}
  label {{ font-size: 0.8rem; display: flex; gap: 0.3rem; align-items: center; }}
  #toolbar {{ position: sticky; top: 0; background: #111; padding: 0.75rem 0; z-index: 10; }}
  button {{ font-size: 0.9rem; padding: 0.5rem 1rem; cursor: pointer; }}
</style>
</head>
<body>
<div id="toolbar">
  <h1>Fish extraction review ({len(entries)} flagged)</h1>
  <p id="instructions">
    Each image below failed the automated single-centered-fish check (reason shown under
    each image; red box is what the detector found, if anything). Check "exclude
    permanently" for any image that's genuinely unusable. Leave unchecked to revisit later
    (e.g. after adjusting detection thresholds and re-running). Click "Export exclusions" -
    it downloads <code>fish_extraction_review_feedback.json</code>. Move it to
    <code>reports/</code>, then re-run:
    <code>python -m fish_extractor.cli --apply-review
    reports/fish_extraction_review_feedback.json</code>
  </p>
  <button id="export">Export exclusions</button>
  <span id="export-status"></span>
</div>
<div class="grid">{"".join(cards)}</div>
<script>
document.getElementById("export").addEventListener("click", () => {{
  const excluded = [];
  document.querySelectorAll(".exclude-box").forEach((box) => {{
    if (box.checked) {{
      excluded.push(box.dataset.key);
    }}
  }});
  const blob = new Blob([JSON.stringify({{excluded}}, null, 2)], {{type: "application/json"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "fish_extraction_review_feedback.json";
  a.click();
  URL.revokeObjectURL(url);
  const status = document.getElementById("export-status");
  status.textContent = ` ${{excluded.length}} image(s) marked for exclusion.`;
}});
</script>
</body>
</html>
"""


def apply_review_feedback(config: PipelineConfig, feedback_path: Path) -> int:
    """Marks reviewer-excluded images as permanently excluded.

    Args:
        config: Pipeline configuration.
        feedback_path: Path to a fish_extraction_review_feedback.json
            exported from the review page's "Export exclusions" button:
            ``{"excluded": ["Genus/Genus_species/003_gbif_3.jpg", ...]}``.

    Returns:
        How many images were newly marked excluded.
    """
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    state = ExtractionState(config.state_path)
    count = 0
    for image_key in feedback.get("excluded", []):
        entry = state.get(image_key)
        if entry.status != "flagged":
            logger.warning(
                "%s is not currently flagged (status=%s), skipping", image_key, entry.status
            )
            continue
        entry.status = "excluded"
        entry.reason = "human_excluded"
        count += 1
    state.save()
    logger.info("Marked %d image(s) as permanently excluded", count)
    return count
