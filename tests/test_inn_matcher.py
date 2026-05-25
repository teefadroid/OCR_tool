"""Tests for INN fuzzy matching."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pharma_ocr.postprocessing.inn_matcher import INNMatcher


class TestINNMatcherFallback:
    def test_loads_fallback_when_no_file(self):
        matcher = INNMatcher(inn_list_path="/nonexistent/path.csv")
        assert matcher._inn_list, "Fallback INN list should be loaded"
        assert matcher._source == "fallback"

    def test_exact_match(self):
        matcher = INNMatcher()
        result = matcher.match("paracetamol")
        assert result["is_valid"]
        assert result["matched"] == "paracetamol"
        assert result["score"] >= 95

    def test_case_insensitive(self):
        matcher = INNMatcher()
        result = matcher.match("PARACETAMOL")
        assert result["is_valid"]

    def test_misspelled_drug_name(self):
        # OCR commonly substitutes 0 for o, etc.
        matcher = INNMatcher(score_cutoff=70)
        result = matcher.match("Paracetam0l")
        assert result["is_valid"]
        assert result["matched"] == "paracetamol"

    def test_unknown_drug(self):
        matcher = INNMatcher()
        result = matcher.match("NonexistentDrugXyz")
        assert not result["is_valid"]
        assert result["matched"] is None

    def test_empty_input(self):
        matcher = INNMatcher()
        result = matcher.match("")
        assert not result["is_valid"]


class TestINNMatcherCustomFile:
    @pytest.fixture
    def custom_csv(self, tmp_path: Path) -> Path:
        p = tmp_path / "custom_inn.csv"
        with p.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["custom_drug_alpha"])
            writer.writerow(["custom_drug_beta"])
            writer.writerow(["custom_drug_gamma"])
        return p

    def test_loads_csv(self, custom_csv: Path):
        matcher = INNMatcher(inn_list_path=custom_csv)
        assert "custom_drug_alpha" in matcher._inn_list

    def test_matches_from_custom_list(self, custom_csv: Path):
        matcher = INNMatcher(inn_list_path=custom_csv, score_cutoff=70)
        result = matcher.match("custom_drug_alpha")
        assert result["is_valid"]
        assert result["matched"] == "custom_drug_alpha"


class TestINNMatchAll:
    def test_finds_multiple_in_text(self):
        matcher = INNMatcher()
        text = "The patient was prescribed Paracetamol 500 mg and Ibuprofen 200 mg"
        results = matcher.match_all(text)
        names = {r["matched"] for r in results}
        assert "paracetamol" in names
        assert "ibuprofen" in names

    def test_deduplicates(self):
        matcher = INNMatcher()
        text = "paracetamol paracetamol paracetamol"
        results = matcher.match_all(text)
        # Same token only matched once
        assert len(results) == 1

    def test_short_tokens_skipped(self):
        # Tokens < 4 chars are filtered out by the regex
        matcher = INNMatcher()
        results = matcher.match_all("a bc def")
        # No real INN match; "def" is 3 chars, also too short by INN standards
        assert all(len(r["matched"] or "") >= 4 for r in results)
