"""
Stage 3: Dual-Model OCR Router

Routes each layout region to the correct OCR model:
  - Arabic regions   -> ArabicGLMClient (Arabic-GLM-OCR-v2)
  - Latin regions    -> GLMOCRClient    (GLM-OCR base)
  - Mixed regions    -> Both models in parallel; higher confidence wins
                        for each field; results merged.

Single-model optimisation:
  When en_model == ar_model (single-model fallback mode), the router
  collapses to one OCR call per region instead of two. This is critical
  for performance since calling the same Ollama model twice in parallel
  serialises internally and just doubles wall-time.

Key design decisions:
  - Parallel execution for mixed regions using ThreadPoolExecutor
  - Confidence-based merging for mixed-script table cells
  - Per-region error isolation: one region failure does not abort the page
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

from ..layout.analyzer import Region
from .arabic_client import ArabicGLMClient
from .glm_client import GLMOCRClient

logger = logging.getLogger(__name__)


class OCRRouter:
    """
    Dual-model OCR router for mixed Arabic/English pharmaceutical documents.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        en_model: str = "glm-ocr",
        ar_model: str = "arabic-glm-ocr",
        max_workers: int = 2,
    ):
        self.en_model = en_model
        self.ar_model = ar_model
        self.single_model_mode = en_model == ar_model
        self.en_client = GLMOCRClient(ollama_url=ollama_url, model=en_model)
        # In single-model mode we still build the Arabic client (for its
        # Arabic-specific prompt + retry fallback) but we point it at the
        # same model name. This keeps behaviour consistent for Arabic
        # regions without making redundant Ollama calls.
        self.ar_client = ArabicGLMClient(ollama_url=ollama_url, model=ar_model)
        self.max_workers = max_workers

        if self.single_model_mode:
            logger.info(
                "Router in single-model mode: '%s' for both languages "
                "(mixed regions will be OCR'd once instead of twice)",
                en_model,
            )

    def process_regions(self, regions: List[Region]) -> List[Region]:
        """
        Process a list of layout regions, assigning OCR text to each.
        Mixed regions are processed with both models in parallel (or once
        if running in single-model mode).

        Returns the same list of regions with .text and .confidence populated.
        """
        latin_regions = [r for r in regions if r.script == "latin"]
        arabic_regions = [r for r in regions if r.script == "arabic"]
        mixed_regions = [r for r in regions if r.script == "mixed"]

        logger.info(
            "Routing %d latin | %d arabic | %d mixed regions",
            len(latin_regions), len(arabic_regions), len(mixed_regions),
        )

        for region in latin_regions:
            self._run_ocr(region, self.en_client, "en")

        for region in arabic_regions:
            self._run_ocr(region, self.ar_client, "ar")

        if mixed_regions:
            if self.single_model_mode:
                # Single model: one call per region, use the Arabic client's
                # logic since it has the Arabic-prompt-then-English-fallback
                # behaviour which works for both scripts.
                for region in mixed_regions:
                    self._run_ocr(region, self.ar_client, "mixed")
            else:
                self._process_mixed_parallel(mixed_regions)

        return regions

    def warmup(self) -> dict:
        """
        Pre-load both models into VRAM. Returns a dict per model.

        In single-model mode, only warms up once. Useful as a first step
        on a fresh Ollama install where cold start can take 60-120 s.
        """
        results = {}
        results[self.en_model] = self.en_client.warmup()
        if not self.single_model_mode:
            results[self.ar_model] = self.ar_client.warmup()
        return results

    def health_check(self) -> dict:
        """Check availability of both models (or just one in single-model mode)."""
        if self.single_model_mode:
            return {self.en_model: self.en_client.health_check()}
        return {
            self.en_model: self.en_client.health_check(),
            self.ar_model: self.ar_client.health_check(),
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_ocr(self, region: Region, client, lang: str) -> None:
        """Run OCR on a region with the given client. Mutates region in place."""
        if region.image is None:
            region.text = ""
            region.confidence = 0.0
            return
        try:
            result = client.ocr(region.image)
            region.text = result.get("text", "")
            region.confidence = result.get("confidence", 0.0)
            region.metadata["ocr_model"] = result.get("model", lang)
            region.metadata["ocr_lang"] = lang
            if result.get("elapsed_s") is not None:
                region.metadata["ocr_elapsed_s"] = round(result["elapsed_s"], 2)
        except Exception as exc:
            logger.error("OCR failed for region %d: %s", region.region_id, exc)
            region.text = ""
            region.confidence = 0.0

    def _process_mixed_parallel(self, regions: List[Region]) -> None:
        """Run both models on each mixed region and merge by confidence."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for region in regions:
                if region.image is None:
                    continue
                f_en = executor.submit(self.en_client.ocr, region.image)
                f_ar = executor.submit(self.ar_client.ocr, region.image)
                futures[region.region_id] = (region, f_en, f_ar)

            # Future timeout: a generous multiple of the client timeout.
            future_timeout = max(self.en_client.timeout, self.ar_client.timeout) + 60

            for region_id, (region, f_en, f_ar) in futures.items():
                try:
                    en_result = f_en.result(timeout=future_timeout)
                    ar_result = f_ar.result(timeout=future_timeout)
                    merged = self._merge_results(en_result, ar_result)
                    region.text = merged["text"]
                    region.confidence = merged["confidence"]
                    region.metadata["ocr_model"] = "dual-merge"
                    region.metadata["en_confidence"] = en_result.get("confidence", 0)
                    region.metadata["ar_confidence"] = ar_result.get("confidence", 0)
                except Exception as exc:
                    logger.error("Parallel OCR failed for region %d: %s", region_id, exc)

    @staticmethod
    def _merge_results(en: dict, ar: dict) -> dict:
        """
        Merge English and Arabic OCR results for a mixed region.
        Strategy: if Arabic result is substantially longer (more content)
        and has decent confidence, prefer Arabic. Otherwise prefer English.
        For truly mixed content, concatenate with BiDi separator.
        """
        en_text = en.get("text", "").strip()
        ar_text = ar.get("text", "").strip()
        en_conf = en.get("confidence", 0.0)
        ar_conf = ar.get("confidence", 0.0)

        if not en_text and not ar_text:
            return {"text": "", "confidence": 0.0}
        if not ar_text:
            return {"text": en_text, "confidence": en_conf}
        if not en_text:
            return {"text": ar_text, "confidence": ar_conf}

        # Both have content: check for Arabic character presence in each
        from ..ingestion.preprocessor import detect_script_in_text
        en_script = detect_script_in_text(en_text)
        ar_script = detect_script_in_text(ar_text)

        # If English model picked up Arabic text too, Arabic model wins
        if ar_script == "arabic" and en_script != "arabic":
            return {"text": ar_text, "confidence": ar_conf}

        # If both have content of different scripts, concatenate
        combined = f"{ar_text}\n\n{en_text}"
        avg_conf = (en_conf + ar_conf) / 2
        return {"text": combined, "confidence": avg_conf}
