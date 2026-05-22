"""Tests for Arabic numeral and unit normalization."""

from __future__ import annotations

from pharma_ocr.postprocessing.numeral_normalizer import (
    extract_strengths,
    normalize_arabic_numerals,
    normalize_arabic_units,
)


class TestNumerals:
    def test_arabic_indic_to_western(self):
        assert normalize_arabic_numerals("٥٠٠") == "500"
        assert normalize_arabic_numerals("١٢٣٤٥٦٧٨٩٠") == "1234567890"

    def test_extended_arabic_indic_to_western(self):
        # Farsi/Urdu numerals
        assert normalize_arabic_numerals("۵۰۰") == "500"

    def test_mixed_text(self):
        text = "Dose: ٥٠٠ مجم twice daily"
        assert normalize_arabic_numerals(text) == "Dose: 500 مجم twice daily"

    def test_unit_misread_mq(self):
        # Common OCR confusion: 'g' read as 'q'
        assert "mg" in normalize_arabic_numerals("500 mq")

    def test_unit_micro_sign(self):
        assert "mcg" in normalize_arabic_numerals("100 µg")

    def test_unit_spacing_inserted(self):
        # "500mg" should become "500 mg"
        result = normalize_arabic_numerals("500mg")
        assert "500 mg" in result

    def test_empty_string(self):
        assert normalize_arabic_numerals("") == ""

    def test_no_arabic_unchanged(self):
        text = "Plain English with 100 mg dose"
        assert normalize_arabic_numerals(text) == text


class TestArabicUnits:
    def test_milligram(self):
        assert normalize_arabic_units("جرعة ٥٠٠ مجم") == "جرعة ٥٠٠ mg"

    def test_milliliter(self):
        assert normalize_arabic_units("١٠٠ مل") == "١٠٠ mL"

    def test_gram(self):
        assert normalize_arabic_units("٢ جم") == "2 g".replace("2", "٢")

    def test_chained_pipeline(self):
        text = "الجرعة ٥٠٠ مجم"
        out = normalize_arabic_units(normalize_arabic_numerals(text))
        assert out == "الجرعة 500 mg"


class TestExtractStrengths:
    def test_single_strength(self):
        results = extract_strengths("Take 500 mg twice daily")
        assert len(results) == 1
        assert results[0]["value"] == "500"
        assert results[0]["unit"] == "mg"

    def test_multiple_strengths(self):
        results = extract_strengths("500 mg paracetamol and 25 mL syrup")
        assert len(results) == 2
        units = {r["unit"] for r in results}
        assert "mg" in units and "ml" in units

    def test_decimal_strength(self):
        results = extract_strengths("0.5 mg dose")
        assert results[0]["value"] == "0.5"

    def test_comma_decimal(self):
        # European decimal notation: 0,5 should be normalized to 0.5
        results = extract_strengths("0,5 mg dose")
        assert results[0]["value"] == "0.5"

    def test_arabic_units_via_pipeline(self):
        # Strengths function calls numeral normalization but not unit translation;
        # the regex only matches Arabic raw "مجم" so verify both paths.
        results = extract_strengths("٥٠٠ مجم")
        assert len(results) == 1
        assert results[0]["value"] == "500"
        # Unit may stay as Arabic since extract_strengths does not run normalize_arabic_units
        assert results[0]["unit"] in ("مجم", "mg")

    def test_no_match(self):
        assert extract_strengths("Just plain text without any doses") == []
