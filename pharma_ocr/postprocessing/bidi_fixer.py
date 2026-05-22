"""
BiDi (Bidirectional) text reordering for mixed Arabic/English output.

Arabic text extracted by OCR models is sometimes returned in visual order
(left-to-right byte sequence) rather than logical order. This module
applies the Unicode Bidirectional Algorithm via python-bidi + arabic-reshaper
to produce correctly ordered, display-ready text.

Also handles:
  - Mixed RTL/LTR lines (e.g. "Paracetamol باراسيتامول 500mg")
  - Paragraphs with inconsistent direction markers
  - Table cells containing both scripts
"""

from __future__ import annotations

import re
import unicodedata

try:
    from bidi.algorithm import get_display
    from arabic_reshaper import reshape
    _BIDI_AVAILABLE = True
except ImportError:
    _BIDI_AVAILABLE = False

# Arabic Unicode ranges used to detect presence of Arabic script
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]")


def fix_bidi_text(text: str, force: bool = False) -> str:
    """
    Apply BiDi reordering to a text string.

    Args:
        text:  Raw OCR output, potentially in wrong visual order.
        force: Apply even if no Arabic characters detected.

    Returns:
        Correctly ordered Unicode string.
    """
    if not text:
        return text

    if not _BIDI_AVAILABLE:
        # Graceful degradation: return as-is with a warning comment
        return text

    has_arabic = bool(_ARABIC_RE.search(text))
    if not has_arabic and not force:
        return text

    # Process line by line to preserve paragraph structure
    lines = text.splitlines(keepends=True)
    fixed_lines = []
    for line in lines:
        stripped = line.rstrip("\n\r")
        if _ARABIC_RE.search(stripped):
            reshaped = reshape(stripped)
            bidi_line = get_display(reshaped)
            fixed_lines.append(bidi_line + line[len(stripped):])
        else:
            fixed_lines.append(line)

    return "".join(fixed_lines)


def has_arabic(text: str) -> bool:
    """Return True if the text contains any Arabic characters."""
    return bool(_ARABIC_RE.search(text))
