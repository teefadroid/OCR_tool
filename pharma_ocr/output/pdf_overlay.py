"""
Searchable PDF Overlay — Stage 5

Adds an invisible text layer to the original scanned PDF, making it
full-text searchable while preserving the original scan appearance.

Uses PyMuPDF (fitz) to overlay extracted text at approximate positions
based on region bounding boxes from the layout analysis stage.

This produces regulatory-archive-grade PDFs where:
  - The original scan is visually intact
  - Ctrl+F / PDF search indexes extracted Arabic and English text
  - Text can be copied and pasted correctly
  - Files are suitable for SFDA/MoH submission archives
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from ..layout.analyzer import Region

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False
    logger.warning("PyMuPDF not installed. PDF overlay disabled. pip install PyMuPDF")


class PDFOverlay:
    """
    Creates a searchable PDF from the original scan + OCR text regions.
    """

    def __init__(self, output_dir: str | Path = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_searchable_pdf(
        self,
        source_pdf_path: str | Path,
        regions_per_page: dict[int, List[Region]],
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Overlay extracted text onto the original PDF scan.

        Args:
            source_pdf_path:    Path to the original scanned PDF.
            regions_per_page:   Dict mapping page_num (1-indexed) → list of Regions
                                with .text and .bbox populated.
            filename:           Output filename. Defaults to <stem>_searchable.pdf

        Returns:
            Path of the created searchable PDF, or None if PyMuPDF unavailable.
        """
        if not _FITZ_AVAILABLE:
            logger.warning("Skipping PDF overlay: PyMuPDF not installed.")
            return None

        source_pdf_path = Path(source_pdf_path)
        stem = source_pdf_path.stem
        out_path = self.output_dir / (filename or f"{stem}_searchable.pdf")

        try:
            doc = fitz.open(str(source_pdf_path))

            for page_idx in range(len(doc)):
                page_num = page_idx + 1
                page = doc[page_idx]
                page_rect = page.rect
                pw, ph = page_rect.width, page_rect.height

                regions = regions_per_page.get(page_num, [])
                for region in regions:
                    if not region.text:
                        continue

                    # Convert pixel bbox to PDF coordinate space
                    x, y, w, h = region.bbox
                    # Regions store pixel coords; we need relative positions
                    # If image was rendered at 300 DPI, 1 PDF point = 300/72 pixels
                    scale = 300 / 72
                    pdf_x0 = x / scale
                    pdf_y0 = y / scale
                    pdf_x1 = (x + w) / scale
                    pdf_y1 = (y + h) / scale

                    # Clamp to page bounds
                    pdf_x0 = max(0, min(pdf_x0, pw))
                    pdf_y0 = max(0, min(pdf_y0, ph))
                    pdf_x1 = max(pdf_x0 + 1, min(pdf_x1, pw))
                    pdf_y1 = max(pdf_y0 + 1, min(pdf_y1, ph))

                    rect = fitz.Rect(pdf_x0, pdf_y0, pdf_x1, pdf_y1)

                    # Insert invisible text block
                    page.insert_textbox(
                        rect,
                        region.text,
                        fontsize=6,
                        color=(1, 1, 1, 0),   # fully transparent
                        render_mode=3,         # invisible (render mode 3)
                        overlay=True,
                    )

            doc.save(str(out_path), garbage=4, deflate=True)
            doc.close()
            logger.info("Searchable PDF written: %s", out_path)
            return out_path

        except Exception as exc:
            logger.error("PDF overlay failed: %s", exc)
            return None

    def merge_images_to_pdf(
        self,
        image_paths: List[str | Path],
        output_filename: str = "merged_scan.pdf",
    ) -> Optional[Path]:
        """
        Combine a list of page images into a single PDF.
        Useful when the input was a folder of TIFF/PNG scans rather than a PDF.
        """
        if not _FITZ_AVAILABLE:
            return None

        out_path = self.output_dir / output_filename
        try:
            doc = fitz.open()
            for img_path in image_paths:
                img_doc = fitz.open(str(img_path))
                doc.insert_pdf(img_doc)
                img_doc.close()
            doc.save(str(out_path))
            doc.close()
            return out_path
        except Exception as exc:
            logger.error("PDF merge failed: %s", exc)
            return None
