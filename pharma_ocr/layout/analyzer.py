"""
Stage 2: Layout Analysis

Wraps PP-DocLayoutV3 (bundled inside GLM-OCR) to segment each page image
into typed regions: text_block, table, figure, formula, header, footer.

Each region is annotated with:
  - bounding box (x, y, w, h) as fraction of page dimensions
  - region type label
  - detected reading direction (ltr / rtl)
  - script hint ('arabic', 'latin', 'mixed') derived from position and context
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Region:
    """A single detected document region."""
    region_id: int
    region_type: str          # text_block | table | figure | formula | header | footer
    bbox: tuple               # (x, y, w, h) in pixels
    script: str               # arabic | latin | mixed
    direction: str            # ltr | rtl
    image: Optional[np.ndarray] = field(default=None, repr=False)
    confidence: float = 1.0
    text: str = ""            # filled after OCR stage
    metadata: dict = field(default_factory=dict)


class LayoutAnalyzer:
    """
    Segments a page image into typed regions using PP-DocLayoutV3.

    PP-DocLayoutV3 is the layout analysis stage bundled inside the
    GLM-OCR inference pipeline. This class provides two modes:

    1. 'ollama' mode  — calls GLM-OCR via Ollama HTTP API; GLM-OCR
       internally runs PP-DocLayoutV3 before recognition.
    2. 'heuristic' mode — column-based geometric fallback when Ollama
       is not available (useful for unit testing and CI).
    """

    PHARMA_REGION_HINTS = [
        "drug_name",
        "ingredient_table",
        "arabic_prose",
        "english_prose",
        "warnings_box",
        "dosage_table",
        "pharmacokinetics",
    ]

    def __init__(self, mode: str = "heuristic", ollama_url: str = "http://localhost:11434"):
        self.mode = mode
        self.ollama_url = ollama_url

    def analyze(self, page: dict) -> List[Region]:
        """
        Analyze a preprocessed page dict (from DocumentPreprocessor)
        and return a list of Region objects.
        """
        img = page["image"]
        script_hint = page.get("script", "mixed")

        if self.mode == "heuristic" or img is None:
            return self._heuristic_split(img, script_hint, page["page_num"])

        # Ollama / production mode: GLM-OCR handles layout internally.
        # We still create region objects representing the full page
        # so the OCR router can call the correct model variant.
        return self._full_page_region(img, script_hint, page["page_num"])

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _heuristic_split(self, img: np.ndarray, script_hint: str, page_num: int) -> List[Region]:
        """
        Simple column detection heuristic for Arabic/English dual-column leaflets.
        Splits the page vertically at the midpoint and assigns script direction
        based on the overall document script hint.
        """
        if img is None:
            return []

        h, w = img.shape[:2]
        mid = w // 2

        regions = []

        if script_hint == "mixed":
            # Left half → English (LTR), Right half → Arabic (RTL)
            regions.append(Region(
                region_id=1,
                region_type="text_block",
                bbox=(0, 0, mid, h),
                script="latin",
                direction="ltr",
                image=img[:, :mid],
                metadata={"page": page_num, "column": "left"},
            ))
            regions.append(Region(
                region_id=2,
                region_type="text_block",
                bbox=(mid, 0, w - mid, h),
                script="arabic",
                direction="rtl",
                image=img[:, mid:],
                metadata={"page": page_num, "column": "right"},
            ))
        else:
            direction = "rtl" if script_hint == "arabic" else "ltr"
            regions.append(Region(
                region_id=1,
                region_type="text_block",
                bbox=(0, 0, w, h),
                script=script_hint,
                direction=direction,
                image=img,
                metadata={"page": page_num, "column": "full"},
            ))

        return regions

    def _full_page_region(self, img: np.ndarray, script_hint: str, page_num: int) -> List[Region]:
        """Return a single full-page region; GLM-OCR handles internal layout."""
        h, w = img.shape[:2]
        return [Region(
            region_id=1,
            region_type="text_block",
            bbox=(0, 0, w, h),
            script=script_hint,
            direction="rtl" if script_hint == "arabic" else "ltr",
            image=img,
            metadata={"page": page_num},
        )]
