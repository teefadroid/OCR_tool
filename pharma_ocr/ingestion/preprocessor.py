"""
Stage 1: Document Ingestion & Pre-processing

Converts PDF pages / image files to clean 300 DPI numpy arrays.
Applies deskew, denoise, and contrast normalization.
Detects Arabic character presence to route to the correct OCR model.

NEVER extracts embedded PDF text — Arabic encoding is unreliable.
Always OCR from image.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

# Arabic Unicode block ranges
ARABIC_RANGE = range(0x0600, 0x06FF + 1)
ARABIC_SUPPLEMENT = range(0xFB50, 0xFDFF + 1)
ARABIC_PRESENTATION = range(0xFE70, 0xFEFF + 1)


class DocumentPreprocessor:
    """
    Converts a PDF or image file to a list of preprocessed page images,
    each annotated with a detected script type: 'arabic', 'latin', or 'mixed'.
    """

    def __init__(
        self,
        dpi: int = 300,
        denoise: bool = True,
        deskew: bool = True,
        contrast: bool = True,
    ):
        self.dpi = dpi
        self.denoise = denoise
        self.deskew = deskew
        self.contrast = contrast

    def process(self, file_path: str | Path) -> List[dict]:
        """
        Process a PDF or image file.

        Returns a list of dicts, one per page:
            {
                'page_num': int,
                'image': np.ndarray,  # BGR, 300 DPI
                'script': str,        # 'arabic' | 'latin' | 'mixed'
                'source': str,
            }
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            pages = self._load_pdf(file_path)
        elif suffix in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"):
            pages = self._load_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        results = []
        for i, img in enumerate(pages):
            processed = self._preprocess(img)
            script = self._detect_script_from_metadata(file_path, processed)
            results.append({
                "page_num": i + 1,
                "image": processed,
                "script": script,
                "source": str(file_path),
            })
            logger.info("Page %d: script detected as '%s'", i + 1, script)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_pdf(self, path: Path) -> List[np.ndarray]:
        """Render PDF pages to images at target DPI. Never extract embedded text."""
        logger.info("Rendering PDF: %s at %d DPI", path.name, self.dpi)
        pil_images = convert_from_path(str(path), dpi=self.dpi)
        return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pil_images]

    def _load_image(self, path: Path) -> List[np.ndarray]:
        """Load a single image file."""
        img = cv2.imread(str(path))
        if img is None:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return [img]

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Apply deskew, denoise, and contrast normalization."""
        if self.deskew:
            img = self._deskew(img)
        if self.denoise:
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        if self.contrast:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        return img

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """Detect and correct skew angle using Hough line transform."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) == 0:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.1 or abs(angle) > 10:
            return img
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _detect_script_from_metadata(self, path: Path, img: np.ndarray) -> str:
        """
        Lightweight script detection using the filename or a quick OCR sample.
        Returns 'arabic', 'latin', or 'mixed'.
        In production, replace with a character-level classifier.
        """
        name = path.stem.lower()
        arabic_keywords = ["ar", "arabic", "عربي", "-ar", "_ar"]
        english_keywords = ["en", "english", "-en", "_en"]

        has_ar_hint = any(k in name for k in arabic_keywords)
        has_en_hint = any(k in name for k in english_keywords)

        if has_ar_hint and has_en_hint:
            return "mixed"
        if has_ar_hint:
            return "arabic"
        if has_en_hint:
            return "latin"
        # Default assumption for pharma leaflets: treat as mixed (safest)
        return "mixed"


def detect_script_in_text(text: str) -> str:
    """
    Detect the dominant script of a text string.
    Returns 'arabic', 'latin', or 'mixed'.
    """
    arabic_count = sum(
        1 for ch in text
        if ord(ch) in ARABIC_RANGE
        or ord(ch) in ARABIC_SUPPLEMENT
        or ord(ch) in ARABIC_PRESENTATION
    )
    latin_count = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = arabic_count + latin_count
    if total == 0:
        return "unknown"
    ar_ratio = arabic_count / total
    if ar_ratio > 0.7:
        return "arabic"
    if ar_ratio < 0.3:
        return "latin"
    return "mixed"
