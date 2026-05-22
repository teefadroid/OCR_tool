"""
Markdown Exporter — Stage 5

Exports PostProcessingResult to a human-readable Markdown file.
Preserves Arabic section headings with RTL Unicode markers.
Suitable for regulatory document archiving and diff tracking.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..postprocessing.pipeline import PostProcessingResult

logger = logging.getLogger(__name__)

_RTL_MARK = "\u200F"   # RIGHT-TO-LEFT MARK
_LTR_MARK = "\u200E"   # LEFT-TO-RIGHT MARK


class MarkdownExporter:
    """
    Exports OCR results to Markdown with bilingual section structure.
    """

    def __init__(self, output_dir: str | Path = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, result: PostProcessingResult, filename: Optional[str] = None) -> Path:
        """Write Markdown file. Returns the output path."""
        content = self._build_markdown(result)
        stem = Path(result.source_file).stem if result.source_file else "ocr_result"
        out_path = self.output_dir / (filename or f"{stem}_ocr.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Markdown written: %s", out_path)
        return out_path

    def to_string(self, result: PostProcessingResult) -> str:
        """Return the Markdown string without writing to disk."""
        return self._build_markdown(result)

    # ------------------------------------------------------------------

    def _build_markdown(self, result: PostProcessingResult) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        source = result.source_file or "unknown"
        lines = [
            f"# PharmOCR Extraction Report",
            f"",
            f"**Source:** `{source}`  ",
            f"**Extracted:** {ts}  ",
            f"**Pages processed:** {result.pages_processed}  ",
            f"**Regions flagged for review:** {len(result.regions_flagged_for_review)}",
            f"",
        ]

        # INN matches
        if result.inn_matches:
            lines += ["## Identified Drug Names (INN Match)", ""]
            for m in result.inn_matches:
                lines.append(
                    f"- `{m['input']}` → **{m['matched']}** (score: {m['score']:.0f}%)"
                )
            lines.append("")

        # Strengths
        if result.strengths:
            lines += ["## Detected Strengths / Doses", ""]
            for s in result.strengths:
                lines.append(f"- {s['value']} {s['unit']}")
            lines.append("")

        # Arabic text
        if result.full_text_arabic:
            lines += [
                f"## {_RTL_MARK}النص العربي (Arabic Content)",
                "",
                "> *Extracted by Arabic-GLM-OCR-v2*",
                "",
                result.full_text_arabic,
                "",
            ]

        # English text
        if result.full_text_english:
            lines += [
                "## English Content",
                "",
                "> *Extracted by GLM-OCR base*",
                "",
                result.full_text_english,
                "",
            ]

        # QC flags
        if result.regions_flagged_for_review:
            lines += [
                "## ⚠️ Regions Requiring Manual Review",
                "",
                f"The following region IDs have confidence below threshold and need human verification:",
                "",
            ]
            for rid in result.regions_flagged_for_review:
                lines.append(f"- Region ID `{rid}`")
            lines.append("")

        return "\n".join(lines)
