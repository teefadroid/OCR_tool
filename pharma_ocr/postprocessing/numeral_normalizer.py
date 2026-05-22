"""
Arabic-Indic and Extended Arabic-Indic numeral normalization.

Pharmaceutical leaflets from GCC/MENA origins frequently use Arabic-Indic
numerals (٠١٢٣٤٥٦٧٨٩) in dosage, strength, and quantity fields.
This module normalizes them to Western Arabic numerals (0-9) for downstream
regulatory field matching and numerical comparison.

Also handles:
  - Extended Arabic-Indic (Farsi/Urdu) numerals: ۰۱۲۳۴۵۶۷۸۹
  - Common OCR mis-reads of dose units (e.g. "mq" → "mg")
  - Strength pattern normalization: "500 MG" → "500 mg"
"""

from __future__ import annotations

import re

# Arabic-Indic → Western Arabic
_AR_IND = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
# Extended Arabic-Indic (Farsi/Urdu) → Western Arabic
_EXT_AR_IND = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Common OCR misreads in dose unit strings
_UNIT_FIXES = [
    (re.compile(r"\bmq\b", re.IGNORECASE), "mg"),
    (re.compile(r"\bmcg\b", re.IGNORECASE), "mcg"),   # keep
    (re.compile(r"\bµg\b"),                 "mcg"),
    (re.compile(r"\biu\b",  re.IGNORECASE), "IU"),
    (re.compile(r"\bml\b",  re.IGNORECASE), "mL"),
    (re.compile(r"\bl\b",   re.IGNORECASE), "L"),
    # Normalize spacing between number and unit: "500mg" → "500 mg"
    (re.compile(r"(\d)(mg|mL|mcg|IU|g|L|%)"), r"\1 \2"),
]

# Strength pattern: capture number + unit combinations
_STRENGTH_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mg|mL|mcg|IU|g|L|%|مجم|مل|جم)",
    re.IGNORECASE,
)


def normalize_arabic_numerals(text: str) -> str:
    """
    Convert Arabic-Indic and Extended Arabic-Indic numerals to Western Arabic.
    Also normalizes dose unit formatting.

    Args:
        text: Raw OCR output string.

    Returns:
        Normalized string with Western Arabic numerals.
    """
    if not text:
        return text

    # Step 1: Translate numeral characters
    text = text.translate(_AR_IND)
    text = text.translate(_EXT_AR_IND)

    # Step 2: Fix common OCR unit misreads
    for pattern, replacement in _UNIT_FIXES:
        text = pattern.sub(replacement, text)

    return text


def extract_strengths(text: str) -> list[dict]:
    """
    Extract all dose/strength values from a text string.

    Returns a list of dicts: [{"value": "500", "unit": "mg", "raw": "500 mg"}, ...]
    """
    normalized = normalize_arabic_numerals(text)
    results = []
    for m in _STRENGTH_RE.finditer(normalized):
        results.append({
            "value": m.group(1).replace(",", "."),
            "unit": m.group(2).lower(),
            "raw": m.group(0),
            "start": m.start(),
            "end": m.end(),
        })
    return results


# Arabic unit suffixes → canonical English
_ARABIC_UNITS = {
    "مجم": "mg",
    "مل":  "mL",
    "جم":  "g",
    "مكجم": "mcg",
    "وحدة دولية": "IU",
}


def normalize_arabic_units(text: str) -> str:
    """Replace Arabic-language unit abbreviations with English equivalents."""
    for ar, en in _ARABIC_UNITS.items():
        text = text.replace(ar, en)
    return text
