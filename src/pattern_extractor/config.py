"""Configuration objects for the pattern-extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClusteringConfig:
    """Settings for the reference-initialized k-means colour clustering.

    Reimplements patternize's patLanK/patRegK method (Van Belleghem et al.
    2018): cluster centres are fit once on a species' reference image, then
    reused as the initial centres for every other image of that species, so
    cluster N means roughly the same colour across images without needing
    pixel-level registration between specimens.

    Attributes:
        k: Number of colour clusters.
        random_seed: Seed for the reference image's initial k-means fit
            (subsequent images are deterministic given that fit's centres).
        max_iterations: Iteration cap per k-means run.
        max_pixels_for_fitting: Cap on how many masked-in pixels the
            iterative k-means fit itself runs on - a real (non-resized)
            fish crop can have millions of masked pixels, and k-means's
            fit cost scales with pixel count x iterations, not just pixel
            count. Above this cap, a random subsample is fit instead;
            every actual pixel is still assigned to its nearest fitted
            centre afterward in a single vectorized pass, so this only
            affects fitting speed, not which pixels get labelled. 50,000
            was checked to reproduce a full 2-million-pixel fit's centres
            to within ~1 RGB unit, in about 1/100th the time.
    """

    k: int = 4
    random_seed: int = 0
    max_iterations: int = 100
    max_pixels_for_fitting: int = 50_000


@dataclass
class ColorConfig:
    """Thresholds for turning cluster fractions into coloring-dimension features.

    Attributes:
        solid_dominant_fraction_threshold: If the largest non-background
            cluster's pixel fraction is at least this high, the fish is
            classified as solid-colored rather than multi-colored.
    """

    solid_dominant_fraction_threshold: float = 0.85


@dataclass
class RegionConfig:
    """Shared thresholds for connected-component region analysis.

    Attributes:
        min_region_area_px: Regions smaller than this are treated as noise
            and ignored by both the spot and stripe extractors.
    """

    min_region_area_px: int = 20


@dataclass
class SpotConfig:
    """Thresholds for classifying regions as spot/freckle-like.

    Attributes:
        max_eccentricity_for_spot: A region's eccentricity (0 = circle, close
            to 1 = elongated line) must be at or below this to count as
            spot-like rather than stripe-like.
        min_spot_count_for_presence: At least this many spot-like regions
            must be found for the image to be called "spotted."
    """

    max_eccentricity_for_spot: float = 0.8
    min_spot_count_for_presence: int = 3


@dataclass
class StripeConfig:
    """Thresholds for classifying regions/texture as stripe-like.

    Attributes:
        min_eccentricity_for_stripe: A region's eccentricity must be at
            least this high to count as elongated/stripe-like.
        min_elongated_region_count: At least this many elongated regions
            must be found for the image to be called "striped" by region
            shape alone.
        min_periodicity_strength: Minimum normalized FFT peak strength
            (of the masked region's intensity profile along its principal
            axis) to call the image "striped" by periodicity, independent
            of the region-count signal above.
    """

    min_eccentricity_for_stripe: float = 0.9
    min_elongated_region_count: int = 2
    min_periodicity_strength: float = 0.3


@dataclass
class PipelineConfig:
    """Top-level configuration for the pattern-extraction pipeline.

    Attributes:
        extracted_root: Root directory of fish_extractor's output (mask-cutout
            crops plus per-image `<stem>_mask.png` files), one subfolder per
            genus/species.
        output_csv_path: Where per-image pattern-feature rows are written.
        reference_filename_stem: Filename stem (without extension) of the
            per-species reference image used to seed clustering.
        clustering: Nested k-means settings.
        color: Nested coloring-dimension thresholds.
        region: Nested shared region-analysis thresholds.
        spot: Nested spot-dimension thresholds.
        stripe: Nested stripe-dimension thresholds.
    """

    extracted_root: Path = field(default_factory=lambda: Path("data/extracted_fish"))
    output_csv_path: Path = field(default_factory=lambda: Path("reports/pattern_features.csv"))
    reference_filename_stem: str = "000_reference"
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    spot: SpotConfig = field(default_factory=SpotConfig)
    stripe: StripeConfig = field(default_factory=StripeConfig)
