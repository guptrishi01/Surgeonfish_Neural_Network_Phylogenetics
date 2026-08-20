"""Reference-initialized k-means colour clustering, reimplemented from
patternize's patLanK/patRegK method (Van Belleghem et al. 2018) - see the
package docstring in __init__.py for the full citation and the two
deliberate departures from the original method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2

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
        otherwise hit routinely.
    """
    pixels = image_rgb[mask].astype(float)
    n_unique_colors = len(np.unique(pixels, axis=0))
    k = min(config.k, n_unique_colors)
    centers, _ = kmeans2(pixels, k, iter=config.max_iterations, minit="++", seed=config.random_seed)
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
    staying identified with) the reference image's cluster centres.

    Args:
        image_rgb: (H, W, 3) uint8 RGB array.
        mask: (H, W) boolean array, True for fish (masked-in) pixels.
        reference_centers: (k, 3) array from `fit_reference()`.
        config: Clustering settings.

    Returns:
        A ClusterResult for this image.
    """
    pixels = image_rgb[mask].astype(float)
    centers, labels = kmeans2(
        pixels, reference_centers.copy(), iter=config.max_iterations, minit="matrix",
        seed=config.random_seed,
    )
    counts = np.bincount(labels, minlength=len(reference_centers))
    fractions = counts / counts.sum() if counts.sum() > 0 else counts.astype(float)
    return ClusterResult(
        centers=centers,
        labels=labels,
        fractions=fractions,
        mask_shape=mask.shape,
        mask_coords=np.where(mask),
    )
