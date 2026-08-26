# CLAUDE.md

Project-specific context for the Surgeonfish Visual Phenomics and Phylogenetic Inference project.

**Document map** (all planned phases are complete as of v6.1.0 — see the Version Control section for what a further change would be versioned as):
- [README.md](README.md) — research question, results, pipeline, how to run it, limitations.
- [METHODS.md](METHODS.md) — statistical design and preregistration (why Kmult is primary, the admissibility transforms, integrity checks).
- [CHANGELOG.md](CHANGELOG.md) — full version history. **The changelog lives here, not in README.md.**
- [BACKGROUND.md](BACKGROUND.md) — pattern-genetics literature review.

## Version Control

Before each implementation/commit, ask whether it's a **major**, **minor**, or **patch** change:

- **Major** — completion of a full rebuild phase. All planned phases are now done (0 data collection → 1 fish extraction → 2 pattern extraction → 3 distance matrices → 4 phylogenetic comparison → 5 documentation), so a further major bump would mean a genuinely new phase, not more work on an existing one. Phase 6 (genomics) was scoped and deliberately declined on power grounds — see README.md's Status section before reviving it.
- **Minor** — a fix or discrete step within a phase (e.g. adding the review-page loop, fixing the numbering bug, adding retry logic).
- **Patch** — a bug fix that doesn't add a step, just corrects one.

Version format is `major.minor.patch`. Phase 0's completion is `1.0.0`; each subsequent phase's completion bumps the major version and resets minor/patch to zero. Record the current version in README.md's header **and** a changelog entry at the top of [CHANGELOG.md](CHANGELOG.md) whenever it changes — the implementation isn't done until that's recorded.

## Data capture

Images are sourced programmatically via `src/dataset_builder/` (see its module docstrings for implementation detail) — not hand-collected, and not scraped from arbitrary web search results.

- **Source: GBIF, not iNaturalist's own API.** GBIF's `occurrence/search` aggregates iNaturalist's research-grade CC-licensed photos (plus other providers), so coverage is comparable. iNaturalist's own API host's `robots.txt` disallows generic query-string access (`Disallow: /*?`) for `User-agent: *`; GBIF's `robots.txt` only blocks an unrelated image-resize-proxy path. Going through GBIF respects that boundary instead of routing around it.
- **robots.txt is a runtime check, not a one-time manual read.** `dataset_builder/robots.py`'s `RobotsChecker` fetches and caches each host's robots.txt and calls `can_fetch()` before every request this pipeline makes — including each image download, which may land on a different host than the API itself (e.g. an S3 bucket). See `tests/dataset_builder/test_robots.py`, which asserts this against the real robots.txt text of both `api.gbif.org` and `api.inaturalist.org`, and proves the check happens before any HTTP call is issued.
- **License allowlist.** Only Creative Commons / public-domain images are accepted (`config.ALLOWED_LICENSE_FRAGMENTS`) — CC0, CC-BY, CC-BY-SA, CC-BY-NC, CC-BY-NC-SA. ND (no-derivatives) variants are deliberately excluded, since this pipeline produces derivative works (segmentation masks, crops, figure overlays). License URL and rights-holder/attribution are logged per image to `reports/image_sourcing_log.csv`.
- **Rate limiting.** One request per second by default (`GBIFClientConfig.request_delay_seconds`), with a descriptive `User-Agent` identifying the project and a contact address.
- **Automated filters catch mechanical problems only.** Resolution, aspect ratio, near-duplicate hashing, and border-region busyness are enforced automatically (`quality_filter.py`). A pilot run showed that "is this fish actually close-up with a clean background" is *not* reliably decidable from generic image statistics — two different heuristics tried (global edge density, center-crop sharpness) each disagreed with visual judgment, in one case flagging the best photo in a batch as bad. That judgment call is left to a human.
- **Human review loop, not more heuristic tuning.** `dataset_builder/review.py` generates `reports/review.html` — a grid of every species' current images with keep/reject checkboxes — after each collection run. Exporting produces `review_feedback.json`; `python -m dataset_builder.cli --apply-review reports/review_feedback.json` deletes the rejected images, permanently blocks their GBIF occurrence keys from being re-offered, and backfills each affected species back toward target before regenerating the review page. Expect to run this loop more than once per species.
- **Resumable, idempotent state.** `dataset_builder/state.py` persists per-species progress (accepted images, seen/rejected occurrence keys, exhaustion) to `data/raw_images_state.json`, so re-running the CLI never re-downloads or re-evaluates a candidate already seen in a prior run.

## Testing

**Backend:** pytest for unit tests of each pipeline stage (segmentation evaluation, feature extraction, distance matrix construction, Kmult/Mantel phylogenetic-signal tests). Run with coverage (`pytest --cov`) — treat coverage gaps as a signal to look for untested *or redundant* paths, not just a number to push up.

**Phase gate:** run the full `pytest` suite after implementing each rebuild phase, before considering that phase done — not just once at the end. This project has already shipped one silently-broken metric; catching a regression at the phase boundary it was introduced in, rather than several phases later, is the entire point.

**No guessing on unfamiliar library/API surfaces — verify field names, argument names, and return shapes before writing code that depends on them, not after.** This happened for real: `src/r/phase4_kmult.R`'s first draft called `sapply(physignal_results, function(r) r$P.value)` on `geomorph::physignal.z()`'s return object without ever confirming that field name against real output. Steps 0–4 ran correctly — including several real minutes of computation — before Step 5 crashed on a wrong guess (`P.value` isn't the actual field name). Before writing code that depends on an unfamiliar function's exact return shape or argument names, confirm it against real documentation/source, or a small throwaway call whose actual output gets inspected (`str()`, `dir()`, `print()`, whatever the language offers) — not a plausible-looking guess carried straight into the real implementation. Applies most acutely to R/`geomorph` calls, since no R runtime is available in this project's usual local working environment to check interfaces the way Python ones can be checked directly — flag that limitation explicitly rather than guessing past it.

There is no frontend in this project — it is a command-line research pipeline, not an application with a UI.

## Code conventions

Python for the entire pipeline (fish detection/segmentation, pattern feature extraction, distance matrices, phylogenetic statistics). GPU-dependent stages (currently `fish_extractor`'s Grounded SAM 2 detection/segmentation) run on Google Colab, not a local machine or HPC/SLURM cluster — code is written as plain importable Python (testable locally on CPU-only logic with the model calls mocked out) so the same functions run identically from a script or a Colab cell.

- **Dataclass-based configuration** — pipeline stage parameters (detection/segmentation thresholds, QA-gate thresholds, feature-extraction settings, Kmult/Mantel permutation counts) as `@dataclass`, not raw module-level constants or dicts.
- **Pipeline classes with incremental state tracking** — each stage (collect data → identify/extract fish → extract pattern features → build distance matrices → compare to phylogeny) should track what it has already processed, so re-running a stage doesn't silently reprocess or duplicate outputs. Exception: stages that are fast, deterministic, and local (no network/GPU calls) may skip this and simply recompute in full on re-run, if documented — see `pattern_extractor/pipeline.py`'s module docstring for the reasoning.
- **Per-module logging** — `logging.getLogger(__name__)` per module, not a shared root logger.
- **Docstrings** — every module, class, and function gets a Google-style docstring (`Args:` / `Returns:`, plus `Raises:` where relevant). This matters especially for anything computing a metric or statistic — the docstring is what keeps the *actual* definition of a number legible, given this pipeline has already had one metric silently mean something other than its name claimed.
- **Ruff**: rules `E, F, I, W`, 100-char line limit.
