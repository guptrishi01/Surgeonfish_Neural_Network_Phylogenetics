"""Reference-initialized k-means colour clustering, reimplemented from
patternize's patLanK/patRegK method (Van Belleghem et al. 2018) - see the
package docstring in __init__.py for the full citation and the two
deliberate departures from the original method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2, vq

from pattern_extractor.config import ClusteringConfig

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Per-image cluster assignment output.

    Attributes:
        centers: (k, 3) array of this image's fitted cluster-centre RGB
            values (starts from, but may drift from, the reference centres).
        labels: 1D array of cluster index per masked-in pixel, in the same
            order as `mask_coords`.
        fractions: (k,) array - each cluster's share of masked-in pixels.
        mask_shape: (height, width) of the source image.
        mask_coords: (ys, xs) coordinate arrays of the masked-in pixels.
    """

    centers: np.ndarray
    labels: np.ndarray
    fractions: np.ndarray
    mask_shape: tuple[int, int]
    mask_coords: tuple[np.ndarray, np.ndarray]

    def label_image(self) -> np.ndarray:
        """Scatters `labels` back into a 2D array, -1 outside the mask."""
        image = np.full(self.mask_shape, -1, dtype=int)
        image[self.mask_coords] = self.labels
        return image

    def binary_mask_for_cluster(self, cluster_index: int) -> np.ndarray:
        """2D boolean array, True where a pixel belongs to `cluster_index`."""
        return self.label_image() == cluster_index


def _subsample(pixels: np.ndarray, max_pixels: int, seed: int) -> np.ndarray:
    """Randomly subsamples rows of `pixels` down to `max_pixels`, if larger.

    `fish_extractor` deliberately doesn't resize crops (native resolution),
    so a real fish mask can be hundreds of thousands to millions of pixels -
    feeding all of them into k-means's iterative fit (up to
    ``max_iterations`` passes over every point) scales badly and was
    measured taking tens of seconds per image at real crop sizes, times
    ~900 images. A random subsample is standard practice for colour
    clustering: centres fit from a representative sample converge to
    within ~1 RGB unit of the full-data fit at 50,000 samples (checked
    against a 2-million-pixel synthetic image), in roughly 1/100th the
    time. Every synthetic test image used before this was small enough
    (under ~1,200 pixels) that this cliff was never exercised.
    """
    if len(pixels) <= max_pixels:
        return pixels
    rng = np.random.default_rng(seed)
    return pixels[rng.choice(len(pixels), max_pixels, replace=False)]


def fit_reference(image_rgb: np.ndarray, mask: np.ndarray, config: ClusteringConfig) -> np.ndarray:
    """Fits k-means on a species' reference image, returning cluster centres.

    Args:
        image_rgb: (H, W, 3) uint8 RGB array.
        mask: (H, W) boolean array, True for fish (masked-in) pixels.
        config: Clustering settings.

    Returns:
        (k, 3) float array of cluster-centre RGB values, to be reused as
        the initial centres for every other image of this species. k is
        capped at the reference image's number of distinct colours (not
        just its pixel count) - asking k-means for more clusters than
        there are distinct colours produces duplicate/empty clusters and
        scipy warnings, which a solid-coloured reference fish would
        otherwise hit routinely. The fit itself runs on at most
        ``config.max_pixels_for_fitting`` randomly-sampled pixels, not
        every masked-in pixel - see `_subsample`.
    """
    pixels = image_rgb[mask].astype(float)
    n_unique_colors = len(np.unique(pixels, axis=0))
    k = min(config.k, n_unique_colors)
    fit_pixels = _subsample(pixels, config.max_pixels_for_fitting, config.random_seed)
    centers, _ = kmeans2(
        fit_pixels, k, iter=config.max_iterations, minit="++", seed=config.random_seed
    )
    return centers


def assign_clusters(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    reference_centers: np.ndarray,
    config: ClusteringConfig,
) -> ClusterResult:
    """Clusters an image's masked-in pixels, starting from reference centres.

    Per patternize's method: reference_centers seed the k-means fit rather
    than being used as fixed nearest-centroid boundaries, so this image's
    clusters can adapt to its own lighting/colour while starting from (and
    staying identified with) the reference image's cluster centres. The
    iterative fit runs on at most ``config.max_pixels_for_fitting`` randomly
    -sampled pixels (see `_subsample`); every actual masked-in pixel is
    then assigned to its nearest fitted centre in a single vectorized pass
    (`scipy.cluster.vq.vq`, not iterative), so `fractions`/`label_image()`
    still reflect the whole image, not just the fitting sample.

    Args:
        image_rgb: (H, W, 3) uint8 RGB array.
        mask: (H, W) boolean array, True for fish (masked-in) pixels.
        reference_centers: (k, 3) array from `fit_reference()`.
        config: Clustering settings.

    Returns:
        A ClusterResult for this image.
    """
    pixels = image_rgb[mask].astype(float)
    fit_pixels = _subsample(pixels, config.max_pixels_for_fitting, config.random_seed)
    centers, _ = kmeans2(
        fit_pixels, reference_centers.copy(), iter=config.max_iterations, minit="matrix",
        seed=config.random_seed,
    )
    labels, _ = vq(pixels, centers)
    counts = np.bincount(labels, minlength=len(reference_centers))
    fractions = counts / counts.sum() if counts.sum() > 0 else counts.astype(float)
    return ClusterResult(
        centers=centers,
        labels=labels,
        fractions=fractions,
        mask_shape=mask.shape,
        mask_coords=np.where(mask),
    )
