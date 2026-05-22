"""
PharmOCR — Mixed Arabic/English Pharmaceutical Leaflet OCR Pipeline

A production-grade, locally-deployed OCR pipeline for pharmaceutical
package inserts and regulatory documents.

Models:
  - zai-org/GLM-OCR        (English / Latin regions)
  - Arabic-GLM-OCR-v2      (Arabic regions)
  - PP-DocLayoutV3         (layout analysis, bundled with GLM-OCR)

Pipeline stages:
  1. ingestion     — PDF/image → 300 DPI numpy arrays (never trusts embedded PDF text)
  2. layout        — PP-DocLayoutV3 region segmentation
  3. ocr           — Dual-model routing (EN + AR parallel)
  4. postprocessing— BiDi fix, numeral normalization, INN matching
  5. output        — JSON (SFDA schema), Markdown, Searchable PDF
  6. api           — FastAPI service for Next.js integration
"""

__version__ = "1.0.0"
__author__ = "PharmOCR"
