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
        max_hue_dispersion_for_solid: An image is classified solid-coloured
            if the mean distance of every masked-in pixel's hue/saturation
            vector (see color_space.rgb_to_hue_sat_vector) from their
            collective mean is at or below this value. Replaces a prior
            "does one of k forced-partition k-means clusters cover >=85% of
            the body" check (`solid_dominant_fraction_threshold`), which
            the manual validation split (v2.2.0) showed was badly broken:
            only 3% of a 155-image hand-labeled sample was classified
            solid, even though 60% genuinely looked solid-coloured.
            Root cause, confirmed by testing directly against real photos:
            k-means asked for k=4 clusters will partition a genuinely
            unimodal colour distribution into k similarly-sized pieces
            regardless of how tightly clustered the real data is - there's
            no mechanism for it to report "these are all basically one
            colour." This dispersion measure sidesteps that by not forcing
            a k-way partition at all - it's a direct spread statistic over
            every pixel. 0.20 was empirically chosen, not guessed: swept
            across the same 155-image hand-labeled sample (masks
            approximated via a corner flood-fill, not real SAM2 masks, so
            production accuracy should be checked rather than assumed
            identical), it gave the best accuracy found (71%, up from the
            old check's 41% - worse than a trivial majority-class baseline)
            at precision 72% / recall 84%. Real remaining disagreement is
            expected, not a bug: some fish genuinely have one dominant
            colour plus a small accent patch, a case reasonable people
            would label differently.
    """

    max_hue_dispersion_for_solid: float = 0.20


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

    All four thresholds below were empirically calibrated (v2.2.1) against
    the 155-image manually-labeled validation sample, via a grid search
    maximizing F1 score - replacing the "reasoned interim value" defaults
    used through v2.1.1, none of which had been checked against real
    ground truth yet. Important caveat carried by all of them: calibration
    used masks approximated by a corner flood-fill (the real images'
    grey background is contiguous with the border in fish_extractor's
    output), not real SAM2 masks - production accuracy on the real
    pipeline should be re-checked via the same notebook validation loop,
    not assumed identical.

    Attributes:
        min_eccentricity_for_stripe: A region's eccentricity must be at
            least this high to count as elongated/stripe-like. 0.97
            (up from 0.9) - the calibration grid found higher eccentricity
            cutoffs consistently scored better, meaning many real
            false-positive regions (fin rays, shading edges) were
            elongated but not *extremely* elongated.
        min_elongated_region_count: At least this many elongated regions
            must be found for the image to be called "striped" by region
            shape alone. 20 (up from 2) - the single biggest change this
            round. Real photos routinely have a handful of small,
            genuinely elongated-and-narrow regions from fin rays, JPEG
            noise, and mask-boundary artifacts even on solid-coloured
            fish; only genuinely multi-stripe patterns (confirmed via the
            v2.0.5 diagnostic: Acanthurus lineatus showed ~30 real stripe
            regions) accumulate this many.
        max_stripe_width_fraction: A region's minor-axis width must be at
            most this fraction of sqrt(total masked pixels) - a proxy for
            the fish's characteristic size - to count as stripe-like.
            0.12 (up from 0.08); the calibration grid found this looser
            width allowance worked better paired with the much stricter
            eccentricity/count thresholds above. Known residual confound
            this doesn't fully fix: fin rays are genuinely thin and
            elongated, so some can still pass even a strict check - the
            higher min_elongated_region_count above is what mainly
            absorbs that now, not this threshold alone.
        min_periodicity_strength: Minimum normalized FFT peak strength
            (of the masked region's intensity profile along its principal
            axis) to call the image "striped" by periodicity, independent
            of the region-count signal above. Effectively disabled (0.9,
            unreachable in practice - observed real values ranged
            ~0.04-0.49) after calibration showed essentially zero
            separation between genuinely striped and non-striped real
            images at every min_periodicity_cycles value tested (e.g. at
            cycles=3: striped mean 0.086 vs. non-striped mean 0.084) -
            worse than the v2.1.1 fix intended, the whole periodicity
            approach carries no real signal on real photos, at least
            under this calibration's approximate masks. Not deleted,
            since real SAM2 masks (precise mask boundaries, not a crude
            flood-fill approximation) might behave differently - this is
            flagged as needing re-validation, not concluded dead.
        min_periodicity_cycles: Minimum number of repeats a frequency must
            represent across the profile to be eligible as the "peak" used
            for min_periodicity_strength. Kept at 3 (the v2.1.1 value) -
            moot while min_periodicity_strength is effectively disabled,
            but left in place for when periodicity is re-evaluated against
            real masks.
    """

    min_eccentricity_for_stripe: float = 0.97
    min_elongated_region_count: int = 20
    max_stripe_width_fraction: float = 0.12
    min_periodicity_strength: float = 0.9
    min_periodicity_cycles: int = 3


@dataclass
class ValidationConfig:
    """Settings for the manually-labeled validation split.

    Not a train/test split for a learned model - every pattern_extractor
    feature is classical CV (k-means clustering, region geometry, FFT
    periodicity) with no learned parameters to overfit. This exists to
    answer a narrower question: do the extracted features actually track
    what a human would call each image's pattern, checked against a sample
    the extraction thresholds weren't tuned against by eye? Built after
    three threshold fixes in a row (RegionConfig.min_region_area_fraction,
    StripeConfig.max_stripe_width_fraction, StripeConfig.
    min_periodicity_cycles) each found a genuinely different real confound
    (a fixed noise floor, wide shading regions, an inverted periodicity
    signal) without fully closing the gap - two further confounds (fin-ray
    anatomy, image blur/noise texture) were still visible in the same
    diagnostic that found the first three, which is exactly the class of
    judgment call this project's own dataset_builder/quality_filter.py
    already learned needs a human, not another geometric heuristic (see
    CLAUDE.md's Data capture section).

    Attributes:
        sample_fraction: Target fraction of each species' non-reference
            images to include in the manual-labeling sample.
        min_per_species: Minimum images sampled per species, even if
            sample_fraction would round to 0 for a sparse species - every
            species gets at least one manually-labeled data point.
        random_seed: Seed for sample selection, so re-running reproduces
            the same sample instead of a new one each time.
        labeling_html_path: Where the self-contained labeling page is
            written.
        labels_json_path: Where the manual labels exported from the
            labeling page's "Export labels" button are expected once
            downloaded and placed here.
        report_csv_path: Where the per-image comparison report (extracted
            features vs. manual labels) is written.
    """

    sample_fraction: float = 0.2
    min_per_species: int = 1
    random_seed: int = 0
    labeling_html_path: Path = field(
        default_factory=lambda: Path("reports/pattern_validation_labeling.html")
    )
    labels_json_path: Path = field(
        default_factory=lambda: Path("reports/pattern_validation_labels.json")
    )
    report_csv_path: Path = field(
        default_factory=lambda: Path("reports/pattern_validation_report.csv")
    )


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
