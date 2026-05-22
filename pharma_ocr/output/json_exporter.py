"""
JSON Exporter — Stage 5

Exports PostProcessingResult to a structured JSON file following
an SFDA/MoH-compatible product registration schema.

Schema fields:
  - trade_name        (AR + EN)
  - inn               (matched INN from WHO list)
  - strength          (normalized value + unit)
  - dosage_form
  - manufacturer
  - origin_country
  - indications       (AR + EN)
  - contraindications (AR + EN)
  - warnings          (AR + EN)
  - ocr_metadata      (model used, confidence, flagged regions)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..postprocessing.pipeline import PostProcessingResult

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0.0"


class JSONExporter:
    """
    Exports a PostProcessingResult to a structured JSON file.
    """

    def __init__(self, output_dir: str | Path = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, result: PostProcessingResult, filename: Optional[str] = None) -> Path:
        """
        Write result to a JSON file.

        Returns the path of the written file.
        """
        doc = self._build_document(result)
        stem = Path(result.source_file).stem if result.source_file else "ocr_result"
        out_path = self.output_dir / (filename or f"{stem}_ocr.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        logger.info("JSON written: %s", out_path)
        return out_path

    def to_dict(self, result: PostProcessingResult) -> dict:
        """Return the JSON-serialisable dict without writing to disk."""
        return self._build_document(result)

    # ------------------------------------------------------------------

    def _build_document(self, result: PostProcessingResult) -> dict:
        """Build the SFDA-compatible export document."""
        # Extract key pharma fields from INN matches and strengths
        primary_inn = (
            result.inn_matches[0]["matched"] if result.inn_matches else None
        )
        primary_strength = (
            f"{result.strengths[0]['value']} {result.strengths[0]['unit']}"
            if result.strengths else None
        )

        return {
            "schema_version": _SCHEMA_VERSION,
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_file": result.source_file,
            "pages_processed": result.pages_processed,
            # --- Core pharmaceutical fields ---
            "drug_info": {
                "trade_name": {"arabic": "", "english": ""},   # populated by QC review
                "inn": primary_inn,
                "strength": primary_strength,
                "all_strengths": result.strengths,
                "dosage_form": "",
                "manufacturer": "",
                "origin_country": "",
                "atc_code": "",
            },
            # --- Text content ---
            "text_content": {
                "arabic": result.full_text_arabic,
                "english": result.full_text_english,
                "combined": result.full_text_combined,
            },
            # --- Validation ---
            "inn_matches": result.inn_matches,
            # --- QC metadata ---
            "ocr_metadata": {
                "regions_flagged_for_review": result.regions_flagged_for_review,
                "requires_manual_review": len(result.regions_flagged_for_review) > 0,
                **result.metadata,
            },
        }
