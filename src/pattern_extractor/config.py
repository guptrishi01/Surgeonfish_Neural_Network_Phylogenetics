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
        min_region_area_fraction: Regions smaller than this fraction of the
            image's total masked-in (fish) pixel count are treated as noise
            and ignored by both the spot and stripe extractors. Was
            previously a fixed 20-pixel absolute count, calibrated against
            tiny synthetic test images (under ~1,200 pixels) and never
            exercised at real crop sizes - fish_extractor doesn't resize
            crops, so masked area ranges from tens of thousands to millions
            of pixels, and a fixed pixel floor doesn't scale with that.
            Measured against real Phase 1/2 output: with the old 20px
            floor, 99% of all 856 real images were flagged
            `stripe_present` regardless of the species' actual pattern
            (including species with no real stripes at all) - hundreds of
            small, incidentally-elongated noise/texture regions (lighting
            variation, blended pixels at the mask boundary, JPEG artifacts)
            were clearing that floor easily. 0.001 (0.1% of the fish's
            visible area) is a reasoned interim default, not yet validated
            against manually-labeled ground truth - the planned but
            not-yet-built 60/20/20 validation split (see Planned Approach
            step 2) is the right way to calibrate this properly.
    """

    min_region_area_fraction: float = 0.001


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
        max_stripe_width_fraction: A region's minor-axis width must be at
            most this fraction of sqrt(total masked pixels) - a proxy for
            the fish's characteristic size - to count as stripe-like.
            Added after a diagnostic visualization against real Phase 1/2
            output showed eccentricity alone couldn't tell a genuine thin
            stripe band apart from one wide, smooth lighting/shading zone
            spanning half a fish's body: both score high eccentricity
            since the fish silhouette itself is elongated, so any region
            covering roughly half of it inherits that elongation without
            being a stripe. 0.08 is a reasoned interim value (a real
            stripe band should be a small fraction of the fish's size,
            not comparable to half the body) - not yet validated against
            manually-labeled ground truth, same caveat as
            RegionConfig.min_region_area_fraction. Known residual
            confound this doesn't fix: fin rays are genuinely thin and
            elongated, so they can still pass this check even though
            they're fin anatomy, not a body colour pattern - the planned
            60/20/20 validation split (see Planned Approach step 2) is
            needed to catch that class of false positive.
        min_periodicity_strength: Minimum normalized FFT peak strength
            (of the masked region's intensity profile along its principal
            axis) to call the image "striped" by periodicity, independent
            of the region-count signal above.
        min_periodicity_cycles: Minimum number of repeats a frequency must
            represent across the profile to be eligible as the "peak" used
            for min_periodicity_strength. Added after a real diagnostic
            (Phase 2 notebook step 6b-ii) showed the periodicity signal was
            inverted: genuinely solid-coloured Zebrasoma species (which
            have a real dorsal-to-ventral lighting gradient) scored
            *higher* periodicity than genuinely striped Acanthurus
            lineatus. Cause: a single smooth colour gradient (one lobe, no
            repetition) concentrates almost all of its non-DC spectral
            energy in the lowest available frequency exactly like genuine
            low-count periodicity does - confirmed numerically with a
            synthetic single-lobe gradient scoring 0.673 pre-fix vs. 0.269
            for a genuine 10-stripe pattern. Restricting the eligible peak
            to frequencies representing at least this many repeats across
            the profile excludes that single-lobe case while leaving a
            real repeating pattern's peak untouched. 3 is a reasoned
            interim value (a real stripe pattern should repeat more than
            once or twice to be called "striped" at all) - not yet
            validated against manually-labeled ground truth, same caveat
            as this config's other thresholds.
    """

    min_eccentricity_for_stripe: float = 0.9
    min_elongated_region_count: int = 2
    max_stripe_width_fraction: float = 0.08
    min_periodicity_strength: float = 0.3
    min_periodicity_cycles: int = 3


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
