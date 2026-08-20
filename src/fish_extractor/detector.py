"""Grounding DINO wrapper: zero-shot, text-prompted fish detection.

The only module (besides segmenter.py) that imports torch/transformers -
both are optional dependencies (see pyproject.toml's ``vision`` extra) and
the model is loaded lazily on first use, so importing this module - and
therefore pipeline.py, which imports it - never requires a GPU or these
packages to be installed. Tests mock this class out entirely.

Uses the Hugging Face ``transformers`` port of Grounding DINO rather than
the original IDEA-Research repo, which requires compiling a custom CUDA op -
a common source of friction on Colab.
"""

from __future__ import annotations

import logging

from fish_extractor.config import DetectionConfig
from fish_extractor.qa_gate import Box

logger = logging.getLogger(__name__)


class GroundingDinoDetector:
    """Zero-shot fish detector backed by a Hugging Face Grounding DINO model."""

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Grounding DINO model %s on %s", self._config.model_id, self._device)
        self._processor = AutoProcessor.from_pretrained(self._config.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._config.model_id
        ).to(self._device)

    def detect(self, image: "PILImage") -> list[Box]:  # noqa: F821
        """Runs zero-shot detection for the configured text prompt.

        Args:
            image: A PIL RGB image at native resolution (detection quality
                degrades if the image was pre-downscaled).

        Returns:
            One Box per detected region whose confidence is at least
            ``config.box_threshold`` - not yet filtered by QAGateConfig,
            that happens separately in qa_gate.evaluate().
        """
        self._ensure_loaded()
        import torch

        inputs = self._processor(
            images=image, text=self._config.text_prompt, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        width, height = image.size
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self._config.box_threshold,
            text_threshold=self._config.text_threshold,
            target_sizes=[(height, width)],
        )[0]

        return [
            Box(
                x0=float(box[0]), y0=float(box[1]), x1=float(box[2]), y1=float(box[3]),
                confidence=float(score),
            )
            for box, score in zip(results["boxes"], results["scores"], strict=True)
        ]
