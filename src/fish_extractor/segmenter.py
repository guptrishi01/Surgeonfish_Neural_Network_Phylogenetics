"""SAM 2 wrapper: turns one accepted detection box into a precise mask.

Same lazy-loading, optional-dependency pattern as detector.py - importing
this module never requires torch/sam2 to be installed; only calling
Sam2Segmenter.segment() does.
"""

from __future__ import annotations

import logging

import numpy as np

from fish_extractor.config import SegmentationConfig
from fish_extractor.qa_gate import Box

logger = logging.getLogger(__name__)


class Sam2Segmenter:
    """Box-prompted segmenter backed by a Hugging Face SAM 2.1 checkpoint."""

    def __init__(self, config: SegmentationConfig) -> None:
        self._config = config
        self._predictor = None
        self._device = "cpu"

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # sam2's own build_sam2() defaults device="cuda" unconditionally if
        # not told otherwise, which crashes with AssertionError on a CPU-only
        # torch build/runtime instead of falling back - detect and pass it
        # explicitly, matching GroundingDinoDetector's behavior.
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading SAM 2 model %s on %s", self._config.model_id, self._device)
        self._predictor = SAM2ImagePredictor.from_pretrained(
            self._config.model_id, device=self._device
        )

    def segment(self, image: "PILImage", box: Box) -> np.ndarray:  # noqa: F821
        """Produces a boolean fish/not-fish mask for one image.

        Args:
            image: A PIL RGB image, the same one the box was detected on.
            box: The single QA-gate-qualifying box to prompt SAM 2 with.

        Returns:
            A ``(height, width)`` boolean array, True where the mask covers
            the fish.
        """
        self._ensure_loaded()
        import torch

        with torch.inference_mode():
            self._predictor.set_image(np.array(image.convert("RGB")))
            masks, scores, _ = self._predictor.predict(
                box=np.array([box.x0, box.y0, box.x1, box.y1]),
                multimask_output=False,
            )
        return masks[0].astype(bool)
