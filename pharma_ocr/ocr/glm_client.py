"""
GLM-OCR Base Model Client

Calls zai-org/GLM-OCR via Ollama HTTP API for English/Latin region OCR.
GLM-OCR achieves #1 on OmniDocBench v1.5 (score: 94.62) and supports
tables, formulas (LaTeX), and complex document layouts.

Model: zai-org/GLM-OCR (0.9B parameters)
Backend: Ollama (local, private)
Alt backend: vLLM (see docs/vllm_setup.md for production deployment)

Performance tuning:
  - Set PHARMOCR_OCR_TIMEOUT (seconds) to override the default 300 s timeout.
    First-run cold start of GLM-OCR can take 60-120 s while Ollama loads
    the weights into VRAM; budget accordingly.
  - Set PHARMOCR_MAX_IMAGE_DIM to override the 1600 px long-side cap. The
    vision encoder works on fixed-size patches, so sending huge raw scans
    (3000+ px) mainly costs PNG/JPEG encoding and HTTP transfer time
    without improving accuracy.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from io import BytesIO
from typing import Optional

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to default if unset/invalid."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default


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

    # First-run cold start (model load into VRAM + JIT compile + first inference)
    # is the dominant latency. 300 s covers that on most consumer hardware.
    DEFAULT_TIMEOUT = 300

    # The vision encoder operates on fixed-size image patches; oversized inputs
    # don't improve accuracy but balloon encoding/transfer time.
    DEFAULT_MAX_IMAGE_DIM = 1600

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = DEFAULT_MODEL,
        timeout: Optional[int] = None,
        max_image_dim: Optional[int] = None,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout if timeout is not None else _env_int(
            "PHARMOCR_OCR_TIMEOUT", self.DEFAULT_TIMEOUT
        )
        self.max_image_dim = max_image_dim if max_image_dim is not None else _env_int(
            "PHARMOCR_MAX_IMAGE_DIM", self.DEFAULT_MAX_IMAGE_DIM
        )
        self._api_url = f"{self.ollama_url}/api/generate"

    def ocr(self, image: np.ndarray, prompt: Optional[str] = None) -> dict:
        """
        Run OCR on a single image region.

        Args:
            image: BGR numpy array from OpenCV.
            prompt: Optional custom prompt. Defaults to standard OCR prompt.

        Returns:
            dict with keys:
                'text': str  - extracted text
                'confidence': float  - estimated confidence (0-1)
                'model': str  - model name used
                'raw_response': dict  - full Ollama API response
        """
        image_b64, sent_w, sent_h = self._encode_image(image, self.max_image_dim)
        prompt = prompt or self.DEFAULT_PROMPT

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 4096},
        }

        kb = len(image_b64) * 3 // 4 // 1024
        logger.info(
            "Calling %s (image %dx%d, ~%d KB) with timeout %ds",
            self.model, sent_w, sent_h, kb, self.timeout,
        )
        t0 = time.monotonic()
        try:
            resp = requests.post(self._api_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "").strip()
            elapsed = time.monotonic() - t0
            logger.info(
                "%s returned %d chars in %.1fs", self.model, len(text), elapsed
            )
            return {
                "text": text,
                "confidence": self._estimate_confidence(text),
                "model": self.model,
                "elapsed_s": elapsed,
                "raw_response": data,
            }
        except requests.exceptions.ConnectionError:
            logger.error(
                "Cannot connect to Ollama at %s. Is Ollama running? "
                "Start with: ollama serve", self.ollama_url
            )
            return self._empty_result("connection_error")
        except requests.exceptions.Timeout:
            logger.error(
                "Ollama request to '%s' timed out after %ds. "
                "First-run cold start can take longer than this on slow hardware. "
                "Try 'python -m pharma_ocr.cli warmup' once, or set "
                "PHARMOCR_OCR_TIMEOUT=600 for very slow machines.",
                self.model, self.timeout,
            )
            return self._empty_result("timeout")
        except Exception as exc:
            logger.error("GLM-OCR error: %s", exc)
            return self._empty_result(str(exc))

    def warmup(self) -> bool:
        """
        Force Ollama to load the model into VRAM by sending a tiny request.

        Subsequent requests will skip the cold-start cost. Returns True on
        success, False on timeout/error. Used by the `warmup` CLI command.
        """
        # 32x32 white image - minimal payload, just enough to trigger load.
        white = np.full((32, 32, 3), 255, dtype=np.uint8)
        original_timeout = self.timeout
        # Allow up to 10 minutes for the very first load on slow machines.
        self.timeout = max(self.timeout, 600)
        try:
            logger.info("Warming up '%s' (timeout %ds)...", self.model, self.timeout)
            t0 = time.monotonic()
            result = self.ocr(white, prompt="Extract any visible text.")
            elapsed = time.monotonic() - t0
            ok = "error" not in result
            if ok:
                logger.info("'%s' warmed up in %.1fs", self.model, elapsed)
            else:
                logger.error("Warmup of '%s' failed: %s", self.model, result.get("error"))
            return ok
        finally:
            self.timeout = original_timeout

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
    def _encode_image(img: np.ndarray, max_dim: int) -> tuple[str, int, int]:
        """
        Encode a BGR numpy array to base64.

        - Downscales so the long side <= max_dim (preserves aspect ratio).
        - Uses JPEG (quality 90) for files >50KB; PNG otherwise.
          OCR is robust to mild JPEG artefacts at q=90 and JPEG is 5-10x smaller.

        Returns (base64_string, width_sent, height_sent).
        """
        rgb = img[:, :, ::-1] if img.ndim == 3 else img
        pil = Image.fromarray(rgb)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")

        w, h = pil.size
        if max_dim and max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            pil = pil.resize(new_size, Image.LANCZOS)
            w, h = pil.size

        buffer = BytesIO()
        # Try JPEG first; fall back to PNG only for tiny images where JPEG
        # overhead would actually be larger.
        pil.save(buffer, format="JPEG", quality=90, optimize=True)
        if buffer.tell() < 50_000:  # PNG is competitive below ~50 KB
            buffer = BytesIO()
            pil.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), w, h

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
