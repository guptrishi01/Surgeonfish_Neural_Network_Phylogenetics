"""Manual validation split: sample selection, a hand-labeling page, and a
comparison report against the extracted features.

Not a train/test split for a learned model - every pattern_extractor
feature is classical CV (k-means clustering, region geometry, FFT
periodicity) with no learned parameters to overfit. This exists to check
whether the extracted features actually track what a human would call
each image's pattern - see ValidationConfig's docstring in config.py for
why this was built now rather than another geometric heuristic.

Follows fish_extractor.review's self-contained-HTML-plus-JSON-export
pattern, with one deliberate difference learned from that module's own
history: images are embedded as base64 thumbnails from the start (the
approach fish_extractor's review page only reached after five rounds of
relative-path/Drive-proxy failures - see README.md's v1.3.3-v1.3.7
changelog entries), not linked by relative path.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from pattern_extractor.config import PipelineConfig, ValidationConfig

logger = logging.getLogger(__name__)

_THUMB_MAX_DIM = 480
_THUMB_QUALITY = 80

_FEATURE_DIMENSIONS = ("is_solid", "stripe_present", "spot_present")


def _species_of(image_key: str) -> str:
    return image_key.split("/")[1]


def _read_feature_rows(output_csv_path: Path) -> list[dict]:
    with open(output_csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_sample(
    pipeline_config: PipelineConfig, validation_config: ValidationConfig
) -> list[str]:
    """Deterministically samples a fraction of each species' non-reference images.

    The curated `000_reference` image is excluded from sampling - it's
    already been looked at closely enough to seed clustering, so it's not
    a fair "did the pipeline get this right without human tuning" check.

    Args:
        pipeline_config: Pipeline configuration (for output_csv_path).
        validation_config: Sampling settings.

    Returns:
        Sorted list of sampled image_key strings.
    """
    rows = _read_feature_rows(pipeline_config.output_csv_path)

    by_species: dict[str, list[str]] = {}
    for row in rows:
        if row["is_reference"] == "True":
            continue
        by_species.setdefault(_species_of(row["image_key"]), []).append(row["image_key"])

    rng = random.Random(validation_config.random_seed)
    sample: list[str] = []
    for species in sorted(by_species):
        keys = sorted(by_species[species])
        target = round(len(keys) * validation_config.sample_fraction)
        n = min(max(validation_config.min_per_species, target), len(keys))
        sample.extend(rng.sample(keys, n))

    sample.sort()
    logger.info(
        "Selected %d image(s) across %d species for manual labeling", len(sample), len(by_species)
    )
    return sample


def _embed_thumbnail(image_path: Path) -> str | None:
    if not image_path.exists():
        return None
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((_THUMB_MAX_DIM, _THUMB_MAX_DIM))
            buffer = io.BytesIO()
            im.save(buffer, format="JPEG", quality=_THUMB_QUALITY)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:  # noqa: BLE001
        logger.warning("Could not embed thumbnail for %s", image_path)
        return None


def generate_labeling_html(
    pipeline_config: PipelineConfig,
    validation_config: ValidationConfig,
    sample: list[str],
) -> Path:
    """Builds a self-contained HTML page for hand-labeling the sampled images.

    Each image gets three independent checkboxes - looks solid-coloured /
    looks striped / looks spotted - mirroring the extracted features'
    own non-exclusive structure (a fish can be both striped and spotted)
    rather than one exclusive category, so each checkbox compares directly
    against one extracted boolean. A rough colour-count field is also
    collected for `n_significant_colors`.

    Args:
        pipeline_config: Pipeline configuration (for extracted_root).
        validation_config: Where to write the page.
        sample: image_key strings to label, e.g. from select_sample().

    Returns:
        The path written to.
    """
    output_path = validation_config.labeling_html_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cards = []
    n_embedded = 0
    for image_key in sample:
        image_path = pipeline_config.extracted_root / f"{image_key}.png"
        thumbnail = _embed_thumbnail(image_path)
        if thumbnail is None:
            continue
        n_embedded += 1
        cards.append(f"""
        <div class="image-card">
          <img src="{thumbnail}" loading="lazy">
          <div class="species">{_species_of(image_key)}</div>
          <label><input type="checkbox" class="label-solid" data-key="{image_key}">
            looks solid-coloured</label>
          <label><input type="checkbox" class="label-stripe" data-key="{image_key}">
            looks striped</label>
          <label><input type="checkbox" class="label-spot" data-key="{image_key}">
            looks spotted</label>
          <label class="colors">roughly how many colours:
            <input type="number" class="label-colors" data-key="{image_key}" min="0" step="1">
          </label>
        </div>""")

    labels_filename = validation_config.labels_json_path.name
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Pattern validation labeling</title>
<style>
  body {{
    font-family: system-ui, sans-serif; margin: 0; padding: 1.5rem;
    background: #111; color: #eee;
  }}
  h1 {{ font-size: 1.3rem; }}
  #instructions {{ font-size: 0.8rem; opacity: 0.8; max-width: 60rem; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 0.75rem; }}
  .image-card {{ width: 220px; background: #1c1c1c; border-radius: 6px; padding: 0.4rem; }}
  .image-card img {{
    width: 100%; height: 200px; object-fit: contain; border-radius: 4px; background: #000;
  }}
  .species {{ font-size: 0.75rem; opacity: 0.8; margin: 0.3rem 0; }}
  label {{ font-size: 0.8rem; display: flex; gap: 0.3rem; align-items: center; }}
  label.colors input {{ width: 3rem; }}
  #toolbar {{ position: sticky; top: 0; background: #111; padding: 0.75rem 0; z-index: 10; }}
  button {{ font-size: 0.9rem; padding: 0.5rem 1rem; cursor: pointer; }}
</style>
</head>
<body>
<div id="toolbar">
  <h1>Pattern validation labeling ({n_embedded} image(s))</h1>
  <p id="instructions">
    For each image, check any of "looks solid-coloured" / "looks striped" / "looks spotted"
    that visually apply (a fish can be more than one, or none if it's genuinely irregular in
    some other way), and roughly how many distinct colours you see. Unchecked boxes and blank
    colour counts are recorded as "no"/unset, not skipped - label every image you can. Click
    "Export labels" when done - it downloads <code>{labels_filename}</code>. Move it to
    <code>reports/</code>, then run the comparison-report step.
  </p>
  <button id="export">Export labels</button>
  <span id="export-status"></span>
</div>
<div class="grid">{"".join(cards)}</div>
<script>
document.getElementById("export").addEventListener("click", () => {{
  const stripeByKey = {{}};
  document.querySelectorAll(".label-stripe").forEach((box) => {{
    stripeByKey[box.dataset.key] = box.checked;
  }});
  const spotByKey = {{}};
  document.querySelectorAll(".label-spot").forEach((box) => {{
    spotByKey[box.dataset.key] = box.checked;
  }});
  const colorsByKey = {{}};
  document.querySelectorAll(".label-colors").forEach((input) => {{
    colorsByKey[input.dataset.key] = input.value === "" ? null : parseInt(input.value, 10);
  }});

  const labels = {{}};
  document.querySelectorAll(".label-solid").forEach((box) => {{
    const key = box.dataset.key;
    labels[key] = {{
      is_solid: box.checked,
      stripe_present: stripeByKey[key],
      spot_present: spotByKey[key],
      color_count: colorsByKey[key],
    }};
  }});
  const blob = new Blob([JSON.stringify({{labels}}, null, 2)], {{type: "application/json"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "{labels_filename}";
  a.click();
  URL.revokeObjectURL(url);
  document.getElementById("export-status").textContent =
    ` ${{Object.keys(labels).length}} image(s) labeled.`;
}});
</script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    logger.info(
        "Wrote pattern validation labeling page (%d image(s)) to %s", n_embedded, output_path
    )
    return output_path


def load_labels(labels_json_path: Path) -> dict[str, dict]:
    """Loads manual labels exported from the labeling page.

    Args:
        labels_json_path: Path to the exported JSON
            (``{"labels": {image_key: {...}, ...}}``).

    Returns:
        The image_key -> label dict.
    """
    data = json.loads(labels_json_path.read_text(encoding="utf-8"))
    return data.get("labels", {})


@dataclass
class ComparisonRow:
    """One image's manual label vs. extracted feature, per dimension.

    Attributes:
        image_key: The labeled image.
        species: Parsed from image_key.
        dimension: One of "is_solid", "stripe_present", "spot_present".
        manual: The human label for this dimension.
        extracted: The pipeline's extracted value for this dimension.
        agree: Whether manual and extracted match.
    """

    image_key: str
    species: str
    dimension: str
    manual: bool
    extracted: bool
    agree: bool


def compare_to_features(
    pipeline_config: PipelineConfig, validation_config: ValidationConfig
) -> list[ComparisonRow]:
    """Compares manual labels against pattern_features.csv, per dimension.

    Only images present in both the labels file and the features CSV are
    compared - a labeled image the pipeline never produced a feature row
    for (e.g. it failed extraction) is logged and skipped rather than
    silently treated as a disagreement.

    Args:
        pipeline_config: Pipeline configuration (for output_csv_path).
        validation_config: Where to read labels from and write the report.

    Returns:
        One ComparisonRow per (labeled image, dimension) pair.
    """
    feature_rows = {
        row["image_key"]: row for row in _read_feature_rows(pipeline_config.output_csv_path)
    }
    labels = load_labels(validation_config.labels_json_path)

    rows: list[ComparisonRow] = []
    missing = 0
    for image_key, label in sorted(labels.items()):
        feature_row = feature_rows.get(image_key)
        if feature_row is None:
            missing += 1
            continue
        species = _species_of(image_key)
        for dimension in _FEATURE_DIMENSIONS:
            manual = bool(label.get(dimension, False))
            extracted = feature_row[dimension] == "True"
            rows.append(
                ComparisonRow(
                    image_key=image_key,
                    species=species,
                    dimension=dimension,
                    manual=manual,
                    extracted=extracted,
                    agree=manual == extracted,
                )
            )
    if missing:
        logger.warning("%d labeled image(s) have no matching feature row, skipped", missing)

    _write_report(validation_config.report_csv_path, rows)
    return rows


def _write_report(report_csv_path: Path, rows: list[ComparisonRow]) -> None:
    report_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_key", "species", "dimension", "manual", "extracted", "agree"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_key": row.image_key,
                    "species": row.species,
                    "dimension": row.dimension,
                    "manual": row.manual,
                    "extracted": row.extracted,
                    "agree": row.agree,
                }
            )
    logger.info("Wrote %d comparison row(s) to %s", len(rows), report_csv_path)


def summarize_agreement(rows: list[ComparisonRow]) -> dict[str, float]:
    """Per-dimension agreement rate (fraction of labeled images where manual == extracted).

    Args:
        rows: Output of compare_to_features().

    Returns:
        dimension -> agreement rate in [0, 1]. A dimension with zero
        labeled rows is omitted rather than reported as a misleading 0.0
        or 1.0.
    """
    by_dimension: dict[str, list[bool]] = {}
    for row in rows:
        by_dimension.setdefault(row.dimension, []).append(row.agree)
    return {
        dimension: sum(agreements) / len(agreements)
        for dimension, agreements in by_dimension.items()
        if agreements
    }
