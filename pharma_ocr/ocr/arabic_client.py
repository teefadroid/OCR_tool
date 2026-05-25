"""
Arabic-GLM-OCR Client

Calls sherif1313/Arabic-GLM-OCR-v2 via Ollama for Arabic region OCR.
This model is fine-tuned specifically on Arabic pharmaceutical and
structured documents. It handles:
  - Arabic printed text (high accuracy)
  - Arabic handwriting (good accuracy)
  - Arabic-Indic numerals (converted in post-processing)
  - Mixed Arabic/English within a region

Model: sherif1313/Arabic-GLM-OCR-v2 (fine-tune of zai-org/GLM-OCR)
HuggingFace: https://huggingface.co/sherif1313/Arabic-GLM-OCR-v1
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .glm_client import GLMOCRClient

logger = logging.getLogger(__name__)


class ArabicGLMClient(GLMOCRClient):
    """
    Subclass of GLMOCRClient with Arabic-specific model and prompts.
    Overrides the model name and default prompt for Arabic pharmaceutical text.
    """

    DEFAULT_MODEL = "arabic-glm-ocr"
    DEFAULT_PROMPT = (
        "قم باستخراج جميع النصوص من هذه الصورة بدقة تامة. "
        "حافظ على الجداول بتنسيق Markdown. "
        "حافظ على عناوين الأقسام. "
        "استخرج النص فقط دون أي تعليق إضافي."
    )
    FALLBACK_ENGLISH_PROMPT = (
        "Extract all Arabic text exactly as written from this pharmaceutical document image. "
        "Preserve table structure as Markdown. Preserve Arabic section headings. "
        "Do not translate. Output extracted text only."
    )

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = DEFAULT_MODEL,
        timeout: Optional[int] = None,
        max_image_dim: Optional[int] = None,
        use_english_prompt_fallback: bool = True,
    ):
        super().__init__(
            ollama_url=ollama_url,
            model=model,
            timeout=timeout,
            max_image_dim=max_image_dim,
        )
        self.use_english_prompt_fallback = use_english_prompt_fallback

    def ocr(self, image: np.ndarray, prompt: Optional[str] = None) -> dict:
        """
        Run Arabic OCR. Falls back to English prompt only if the Arabic
        prompt yielded an empty response AND the call did not error out
        (so we don't waste another timeout on a model that's already failing).
        """
        result = super().ocr(image, prompt=prompt or self.DEFAULT_PROMPT)

        if (
            not result.get("text")
            and self.use_english_prompt_fallback
            and not result.get("error")
        ):
            logger.info("Arabic prompt yielded no text; retrying with English fallback prompt")
            result = super().ocr(image, prompt=self.FALLBACK_ENGLISH_PROMPT)
            result["prompt_fallback_used"] = True

        return result

    def health_check(self) -> bool:
        """Check if Arabic model is available. Provides setup guidance if not."""
        available = super().health_check()
        if not available:
            logger.warning(
                "Arabic-GLM-OCR model not found in Ollama. See docs/setup.md "
                "for the three working options (single-model fallback, "
                "HuggingFace transformers, or llama.cpp llama-server)."
            )
        return available
