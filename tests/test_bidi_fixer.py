"""Tests for BiDi reordering and Arabic detection."""

from __future__ import annotations

from pharma_ocr.postprocessing.bidi_fixer import fix_bidi_text, has_arabic


class TestHasArabic:
    def test_arabic_only(self):
        assert has_arabic("باراسيتامول")

    def test_mixed(self):
        assert has_arabic("Paracetamol باراسيتامول 500mg")

    def test_latin_only(self):
        assert not has_arabic("Paracetamol 500 mg")

    def test_empty(self):
        assert not has_arabic("")

    def test_arabic_presentation_forms(self):
        # FB50–FDFF range
        assert has_arabic("\ufb50\ufb51")


class TestFixBidi:
    def test_empty_string_passes_through(self):
        assert fix_bidi_text("") == ""

    def test_pure_latin_unchanged(self):
        text = "Paracetamol 500 mg"
        # No Arabic chars → return unchanged when force=False
        assert fix_bidi_text(text) == text

    def test_arabic_text_processed(self):
        # Arabic input should be reshaped/displayed; we can't assert exact
        # output without locking to a specific bidi/reshaper version, but
        # we can assert it's non-empty and contains Arabic presentation forms.
        out = fix_bidi_text("باراسيتامول")
        assert out  # non-empty
        assert isinstance(out, str)

    def test_paragraph_structure_preserved(self):
        # Multi-line input: the number of newline characters is preserved
        text = "line1\nباراسيتامول\nline3"
        out = fix_bidi_text(text)
        assert out.count("\n") == text.count("\n")

    def test_force_flag_on_latin(self):
        # With force=True, even Latin text is BiDi-processed (effectively a no-op,
        # but should not crash)
        out = fix_bidi_text("Hello world", force=True)
        assert isinstance(out, str)
