"""
GLM-OCR Base Model Client

Calls zai-org/GLM-OCR via Ollama HTTP API for English/Latin region OCR.
GLM-OCR achieves #1 on OmniDocBench v1.5 (score: 94.62) and supports
tables, formulas (LaTeX), and complex document layouts.

Model: zai-org/GLM-OCR (0.9B parameters)
Backend: Ollama (local, private)
Alt backend: vLLM (see docs/vllm_setup.md for production deployment)
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Optional

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)


class GLMOCRClient:
    """
    Client for the GLM-OCR base model via Ollama.

    Handles image encoding, prompt construction, and response parsing.
    """

    DEFAULT_MODEL = "glm-ocr"
    DEFAULT_PROMPT = (
        "You are an expert OCR model. Extract all text from this document image. "
        "Preserve tables as Markdown. Preserve mathematical formulas as LaTeX. "
        "Preserve section headings. Output only the extracted text, nothing else."
    )

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = DEFAULT_MODEL,
        timeout: int = 120,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._api_url = f"{self.ollama_url}/api/generate"

    def ocr(self, image: np.ndarray, prompt: Optional[str] = None) -> dict:
        """
        Run OCR on a single image region.

        Args:
            image: BGR numpy array from OpenCV.
            prompt: Optional custom prompt. Defaults to standard OCR prompt.

        Returns:
            dict with keys:
                'text': str  — extracted text
                'confidence': float  — estimated confidence (0–1)
                'model': str  — model name used
                'raw_response': dict  — full Ollama API response
        """
        image_b64 = self._encode_image(image)
        prompt = prompt or self.DEFAULT_PROMPT

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 4096},
        }

        try:
            resp = requests.post(self._api_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "").strip()
            return {
                "text": text,
                "confidence": self._estimate_confidence(text),
                "model": self.model,
                "raw_response": data,
            }
        except requests.exceptions.ConnectionError:
            logger.error(
                "Cannot connect to Ollama at %s. Is Ollama running? "
                "Start with: ollama serve", self.ollama_url
            )
            return self._empty_result("connection_error")
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out after %ds", self.timeout)
            return self._empty_result("timeout")
        except Exception as exc:
            logger.error("GLM-OCR error: %s", exc)
            return self._empty_result(str(exc))

    def health_check(self) -> bool:
        """Return True if Ollama is reachable and the model is available."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            available = any(self.model in m for m in models)
            if not available:
                logger.warning(
                    "Model '%s' not found in Ollama. Pull with: ollama pull %s",
                    self.model, self.model
                )
            return available
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(img: np.ndarray) -> str:
        """Encode a BGR numpy array to base64 PNG."""
        rgb = img[:, :, ::-1] if img.ndim == 3 else img
        pil = Image.fromarray(rgb)
        buffer = BytesIO()
        pil.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _estimate_confidence(text: str) -> float:
        """Heuristic confidence based on output length and content."""
        if not text:
            return 0.0
        if len(text) < 10:
            return 0.4
        if len(text) > 50:
            return 0.92
        return 0.75

    @staticmethod
    def _empty_result(error: str) -> dict:
        return {"text": "", "confidence": 0.0, "model": "glm-ocr", "error": error, "raw_response": {}}
