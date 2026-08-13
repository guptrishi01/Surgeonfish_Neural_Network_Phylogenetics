"""Configuration objects for the surgeonfish dataset-building pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# License URL fragments that GBIF's `media[].license` field is allowed to match.
# ND (no-derivatives) variants are deliberately excluded: the pipeline produces
# derivative works (segmentation masks, crops, figure overlays), which ND licenses
# don't clearly permit.
ALLOWED_LICENSE_FRAGMENTS: tuple[str, ...] = (
    "publicdomain/zero",
    "publicdomain/mark",
    "licenses/by/",
    "licenses/by-sa/",
    "licenses/by-nc/",
    "licenses/by-nc-sa/",
)


@dataclass
class GBIFClientConfig:
    """Settings for querying the GBIF occurrence API.

    Attributes:
        base_url: GBIF API root.
        page_size: Records requested per paginated search call.
        request_delay_seconds: Minimum delay between requests, for politeness
            and to stay well under GBIF's published rate limits.
        user_agent: Identifies this tool and a contact address, per API
            good-practice norms (GBIF's robots.txt does not restrict the
            occurrence/search or species/match endpoints used here).
        timeout_seconds: Per-request HTTP timeout.
        allowed_license_fragments: Substrings checked against a media record's
            license URL; a record is kept only if one fragment matches.
        max_retries: Retries for transient failures (5xx, connection/timeout
            errors) before giving up on a single request.
        retry_backoff_seconds: Base delay for exponential backoff between
            retries (attempt N waits retry_backoff_seconds * 2**N).
    """

    base_url: str = "https://api.gbif.org/v1"
    page_size: int = 100
    request_delay_seconds: float = 1.0
    user_agent: str = (
        "surgeonfish-phylogenetics-research-bot/0.1 "
        "(UNC Charlotte Dornburg Lab; contact rgupta25@charlotte.edu)"
    )
    timeout_seconds: float = 20.0
    allowed_license_fragments: tuple[str, ...] = ALLOWED_LICENSE_FRAGMENTS
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0


@dataclass
class QualityFilterConfig:
    """Thresholds used to accept or reject a downloaded candidate image.

    Attributes:
        min_short_side_px: Minimum pixel length of the shorter image dimension.
        min_aspect_ratio: Reject images narrower (relatively) than this
            (width/height or height/width, whichever is < 1).
        max_file_size_bytes: Skip downloads larger than this (avoids raw scans
            / multi-page TIFFs miscategorized as StillImage).
        duplicate_hash_hamming_threshold: Two images are treated as
            near-duplicates if their average-hash Hamming distance is at or
            below this value.
        max_border_edge_density: Soft ceiling on edge-pixel fraction within
            the outer border ring, used as a proxy for "busy background."
            Images above this are rejected; images just under it are still
            flagged in the metadata log for the manual spot-check pass.
    """

    min_short_side_px: int = 800
    min_aspect_ratio: float = 0.4
    max_file_size_bytes: int = 25_000_000
    duplicate_hash_hamming_threshold: int = 6
    max_border_edge_density: float = 0.35


@dataclass
class PipelineConfig:
    """Top-level configuration for the dataset-building pipeline.

    Attributes:
        raw_images_root: Root directory holding one subfolder per genus, each
            containing one subfolder per species.
        target_per_species: Balanced image-count goal for every species folder.
        max_candidates_per_species: Safety cap on how many GBIF media records
            to evaluate for a single species before giving up and logging it
            as short of target (prevents an unbounded loop on rare species).
        state_path: Where resumable per-species progress is persisted.
        metadata_csv_path: Where per-image source/license/attribution rows are
            appended, for citation and later manual review.
        gbif: Nested GBIF client settings.
        quality: Nested quality-filter thresholds.
    """

    raw_images_root: Path = field(
        default_factory=lambda: Path("data/raw_images")
    )
    target_per_species: int = 25
    max_candidates_per_species: int = 400
    state_path: Path = field(
        default_factory=lambda: Path("data/raw_images_state.json")
    )
    metadata_csv_path: Path = field(
        default_factory=lambda: Path("reports/image_sourcing_log.csv")
    )
    gbif: GBIFClientConfig = field(default_factory=GBIFClientConfig)
    quality: QualityFilterConfig = field(default_factory=QualityFilterConfig)
