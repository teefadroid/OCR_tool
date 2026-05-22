"""
Post-Processing Pipeline

Orchestrates all post-processing steps on OCR-extracted regions:
  1. Arabic numeral normalization     (٥٠٠ مجم → 500 mg)
  2. Arabic unit translation          (مجم → mg)
  3. BiDi text reordering             (RTL/LTR mixed lines)
  4. INN fuzzy matching               (drug name validation)
  5. Confidence flag for QC review    (regions below threshold)

Returns a PostProcessingResult with structured pharma fields.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ..layout.analyzer import Region
from .bidi_fixer import fix_bidi_text
from .numeral_normalizer import normalize_arabic_numerals, normalize_arabic_units, extract_strengths
from .inn_matcher import INNMatcher

logger = logging.getLogger(__name__)

QC_CONFIDENCE_THRESHOLD = 0.85


@dataclass
class ExtractedField:
    """A single extracted pharmaceutical field."""
    name: str
    value: str
    confidence: float
    source_region_id: int
    needs_review: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class PostProcessingResult:
    """Structured output from the post-processing pipeline for one document."""
    source_file: str
    pages_processed: int
    fields: List[ExtractedField] = field(default_factory=list)
    strengths: List[dict] = field(default_factory=list)
    inn_matches: List[dict] = field(default_factory=list)
    full_text_arabic: str = ""
    full_text_english: str = ""
    full_text_combined: str = ""
    regions_flagged_for_review: List[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "pages_processed": self.pages_processed,
            "fields": [{"name": f.name, "value": f.value, "confidence": f.confidence,
                        "needs_review": f.needs_review} for f in self.fields],
            "strengths": self.strengths,
            "inn_matches": self.inn_matches,
            "full_text_arabic": self.full_text_arabic,
            "full_text_english": self.full_text_english,
            "full_text_combined": self.full_text_combined,
            "regions_flagged_for_review": self.regions_flagged_for_review,
            "metadata": self.metadata,
        }


class PostProcessor:
    """
    Runs the full post-processing pipeline on a list of OCR regions.
    """

    def __init__(
        self,
        inn_list_path: Optional[str] = None,
        confidence_threshold: float = QC_CONFIDENCE_THRESHOLD,
    ):
        self.inn_matcher = INNMatcher(inn_list_path)
        self.confidence_threshold = confidence_threshold

    def process(self, regions: List[Region], source_file: str = "") -> PostProcessingResult:
        """
        Run all post-processing on a list of OCR-annotated regions.
        Regions are mutated in place (text is cleaned); result is returned.
        """
        result = PostProcessingResult(
            source_file=source_file,
            pages_processed=len({r.metadata.get("page", 1) for r in regions}),
        )

        arabic_blocks = []
        english_blocks = []
        all_text_parts = []
        flagged = []

        for region in regions:
            if not region.text:
                continue

            # Step 1 & 2: Normalize numerals and units
            cleaned = normalize_arabic_numerals(region.text)
            cleaned = normalize_arabic_units(cleaned)

            # Step 3: BiDi reordering
            if region.script in ("arabic", "mixed"):
                cleaned = fix_bidi_text(cleaned)

            region.text = cleaned

            # Step 4: QC flag
            if region.confidence < self.confidence_threshold:
                flagged.append(region.region_id)
                region.metadata["needs_review"] = True
                logger.warning(
                    "Region %d flagged for review (confidence: %.2f)",
                    region.region_id, region.confidence
                )

            # Accumulate by script
            if region.script == "arabic":
                arabic_blocks.append(cleaned)
            elif region.script == "latin":
                english_blocks.append(cleaned)
            else:
                arabic_blocks.append(cleaned)
                english_blocks.append(cleaned)

            all_text_parts.append(cleaned)

        result.full_text_arabic = "\n\n".join(arabic_blocks)
        result.full_text_english = "\n\n".join(english_blocks)
        result.full_text_combined = "\n\n".join(all_text_parts)
        result.regions_flagged_for_review = flagged

        # Step 5: INN matching across combined text
        inn_hits = self.inn_matcher.match_all(result.full_text_combined)
        result.inn_matches = inn_hits

        # Step 6: Strength extraction
        result.strengths = extract_strengths(result.full_text_combined)

        logger.info(
            "Post-processing complete: %d INN matches, %d strengths, %d regions flagged",
            len(inn_hits), len(result.strengths), len(flagged)
        )

        return result
