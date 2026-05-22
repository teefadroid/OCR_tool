"""
INN (International Nonproprietary Name) fuzzy matching.

Matches extracted drug names from OCR output against the WHO INN list
and a local pharmaceutical name dictionary.

Uses rapidFuzz for fast approximate string matching — handles:
  - OCR character substitutions (e.g. "Paracetam0l" → "Paracetamol")
  - Case and spacing inconsistencies
  - Arabic transliterations of INN names
  - Partial matches in compound product names
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

try:
    from rapidfuzz import fuzz, process as fuzz_process
    _RAPIDFUZZ = True
except ImportError:
    _RAPIDFUZZ = False

logger = logging.getLogger(__name__)

# Default INN list location (relative to project root)
_DEFAULT_INN_PATH = Path(__file__).parent.parent.parent / "data" / "inn_list" / "who_inn.csv"

# Built-in minimal fallback list of common pharma INNs for offline use
_FALLBACK_INNS = [
    "paracetamol", "acetaminophen", "ibuprofen", "amoxicillin", "metformin",
    "atorvastatin", "omeprazole", "amlodipine", "losartan", "metoprolol",
    "simvastatin", "cetirizine", "loratadine", "aspirin", "diclofenac",
    "pantoprazole", "salbutamol", "albuterol", "dexamethasone", "prednisolone",
    "azithromycin", "ciprofloxacin", "amoxicillin clavulanate", "ceftriaxone",
    "insulin", "metronidazole", "fluconazole", "clarithromycin", "ranitidine",
    "clopidogrel", "warfarin", "enoxaparin", "furosemide", "spironolactone",
    "lisinopril", "enalapril", "ramipril", "valsartan", "bisoprolol",
]


class INNMatcher:
    """
    Fuzzy INN matcher for pharmaceutical drug name validation.

    Usage:
        matcher = INNMatcher()               # loads built-in fallback
        matcher = INNMatcher("who_inn.csv")  # loads full WHO INN list

        result = matcher.match("Paracetam0l")
        # → {"matched": "paracetamol", "score": 95.2, "source": "who_inn"}
    """

    def __init__(self, inn_list_path: Optional[str | Path] = None, score_cutoff: float = 80.0):
        self.score_cutoff = score_cutoff
        self._inn_list: list[str] = []
        self._source = "fallback"
        self._load(inn_list_path)

    def _load(self, path: Optional[str | Path]) -> None:
        """Load INN list from CSV, JSON, or plain text. Falls back to built-in list."""
        target = Path(path) if path else _DEFAULT_INN_PATH

        if target.exists():
            try:
                suffix = target.suffix.lower()
                if suffix == ".csv":
                    with open(target, encoding="utf-8", newline="") as f:
                        reader = csv.reader(f)
                        self._inn_list = [
                            row[0].strip().lower()
                            for row in reader
                            if row and row[0].strip()
                        ]
                elif suffix == ".json":
                    with open(target, encoding="utf-8") as f:
                        data = json.load(f)
                        self._inn_list = [str(x).lower() for x in data]
                else:
                    with open(target, encoding="utf-8") as f:
                        self._inn_list = [line.strip().lower() for line in f if line.strip()]
                self._source = target.name
                logger.info("Loaded %d INNs from %s", len(self._inn_list), target.name)
                return
            except Exception as exc:
                logger.warning("Could not load INN list from %s: %s", target, exc)

        # Fallback
        self._inn_list = list(_FALLBACK_INNS)
        logger.info("Using built-in fallback INN list (%d entries)", len(self._inn_list))

    def match(self, drug_name: str) -> dict:
        """
        Find the best INN match for a drug name string.

        Returns:
            {
                "input": str,
                "matched": str | None,
                "score": float,
                "source": str,
                "is_valid": bool,
            }
        """
        if not drug_name or not drug_name.strip():
            return self._no_match(drug_name)

        query = drug_name.strip().lower()

        if not _RAPIDFUZZ:
            # Fallback: simple substring check
            for inn in self._inn_list:
                if query in inn or inn in query:
                    return {
                        "input": drug_name, "matched": inn,
                        "score": 85.0, "source": self._source, "is_valid": True,
                    }
            return self._no_match(drug_name)

        result = fuzz_process.extractOne(
            query,
            self._inn_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=self.score_cutoff,
        )

        if result:
            matched_inn, score, _ = result
            return {
                "input": drug_name,
                "matched": matched_inn,
                "score": round(score, 1),
                "source": self._source,
                "is_valid": True,
            }
        return self._no_match(drug_name)

    def match_all(self, text: str, candidates: Optional[list[str]] = None) -> list[dict]:
        """
        Find all INN matches in a block of text.
        Splits on whitespace and punctuation, then matches each token.
        """
        import re
        tokens = re.findall(r"[a-zA-Z\u0600-\u06FF]{4,}", text)
        seen = set()
        results = []
        for token in tokens:
            if token.lower() in seen:
                continue
            seen.add(token.lower())
            m = self.match(token)
            if m["is_valid"]:
                results.append(m)
        return results

    @staticmethod
    def _no_match(name: str) -> dict:
        return {
            "input": name, "matched": None,
            "score": 0.0, "source": "none", "is_valid": False,
        }
