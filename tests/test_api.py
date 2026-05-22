"""
Tests for the FastAPI surface with the OCR pipeline mocked out.

These tests do NOT invoke the real OCR models. The module-level
preprocessor / layout / router instances inside `pharma_ocr.api.app`
are monkey-patched so the test runs against the post-processing,
audit log, and HTTP layers only.

Run with:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import io
import os
import sys
import types
from pathlib import Path

import pytest

# We need real numpy for Region.image typing default; conftest.py stubs it
# for the other test modules. Let it lose precedence here.
sys.modules.pop("numpy", None)
try:
    import numpy as np  # type: ignore  # noqa: F401
except ImportError:
    # Fall back to the conftest stub
    np = sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    np.ndarray = type("ndarray", (), {})  # type: ignore[attr-defined]


# Direct outputs to a stable ./test_artifacts dir before importing the API
@pytest.fixture(scope="module", autouse=True)
def _api_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("api")
    os.environ["PHARMOCR_OUTPUT_DIR"] = str(base / "out")
    os.environ["PHARMOCR_AUDIT_DB"] = str(base / "audit.sqlite3")
    yield


@pytest.fixture
def client(monkeypatch):
    """FastAPI TestClient with the pipeline singletons mocked."""
    # Defer the import until the env vars above are set
    from fastapi.testclient import TestClient
    from pharma_ocr.api import app as api_module
    from pharma_ocr.layout.analyzer import Region

    fake_page = {
        "image": _DummyImage(width=800, height=1100),
        "page_num": 1,
        "script": "mixed",
        "dpi": 300,
    }

    def fake_preprocess(input_path):
        return [fake_page]

    def fake_layout(page):
        # Two regions: one English (left col), one Arabic (right col)
        return [
            Region(
                region_id=1, region_type="text_block", bbox=(0, 0, 400, 1100),
                script="latin", direction="ltr", image=None, confidence=0.0,
                text="", metadata={"page": page["page_num"], "column": "left"},
            ),
            Region(
                region_id=2, region_type="text_block", bbox=(400, 0, 400, 1100),
                script="arabic", direction="rtl", image=None, confidence=0.0,
                text="", metadata={"page": page["page_num"], "column": "right"},
            ),
        ]

    def fake_ocr(regions):
        for r in regions:
            if r.script == "latin":
                r.text = (
                    "Paracetamol 500 mg tablet. "
                    "Take 1 tablet twice daily. Contains Ibuprofen 200 mg."
                )
                r.confidence = 0.95
            else:
                r.text = "باراسيتامول ٥٠٠ مجم"
                r.confidence = 0.78  # below default 0.85 → should be flagged
        return regions

    def fake_health():
        return {"glm-ocr": True, "arabic-glm-ocr": True}

    monkeypatch.setattr(api_module._preprocessor, "process", fake_preprocess)
    monkeypatch.setattr(api_module._layout, "analyze", fake_layout)
    monkeypatch.setattr(api_module._router, "process_regions", fake_ocr)
    monkeypatch.setattr(api_module._router, "health_check", fake_health)

    return TestClient(api_module.app)


class _DummyImage:
    """Minimal numpy-ndarray-like stub for layout/output stages we don't exercise."""
    def __init__(self, width: int, height: int):
        self.shape = (height, width, 3)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_ok_when_models_available(self, client):
        r = client.get("/ocr/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["models"] == {"glm-ocr": True, "arabic-glm-ocr": True}

    def test_degraded_when_model_missing(self, client, monkeypatch):
        from pharma_ocr.api import app as api_module
        monkeypatch.setattr(
            api_module._router, "health_check",
            lambda: {"glm-ocr": True, "arabic-glm-ocr": False},
        )
        r = client.get("/ocr/health")
        assert r.status_code == 503
        assert r.json()["status"] == "degraded"


# ---------------------------------------------------------------------------
# Process — full pipeline through mocks
# ---------------------------------------------------------------------------

class TestProcess:
    def _upload(self, client, fmt="json"):
        return client.post(
            "/ocr/process",
            files={"file": ("leaflet.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={"output_format": fmt, "confidence_threshold": "0.85"},
        )

    def test_returns_200_and_job_id(self, client):
        r = self._upload(client)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("job_id")
        assert len(body["job_id"]) >= 8

    def test_inn_matches_extracted(self, client):
        body = self._upload(client).json()
        names = {m["matched"] for m in body["inn_matches"]}
        assert "paracetamol" in names
        assert "ibuprofen" in names

    def test_strengths_extracted(self, client):
        body = self._upload(client).json()
        units = {s["unit"] for s in body["drug_info"]["all_strengths"]}
        assert "mg" in units

    def test_arabic_text_normalized(self, client):
        body = self._upload(client).json()
        # Arabic-Indic '٥٠٠' should have become '500'
        assert "500" in body["text_content"]["combined"]

    def test_low_confidence_region_flagged(self, client):
        body = self._upload(client).json()
        flagged = body["ocr_metadata"]["regions_flagged_for_review"]
        # Region 2 has confidence 0.78 < 0.85 → flagged
        assert 2 in flagged
        assert body["ocr_metadata"]["requires_manual_review"] is True

    def test_unsupported_extension_rejected(self, client):
        r = client.post(
            "/ocr/process",
            files={"file": ("malware.exe", b"MZ", "application/x-msdownload")},
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Result retrieval & job listing
# ---------------------------------------------------------------------------

class TestResultAndJobs:
    def test_result_retrieval_round_trip(self, client):
        r = client.post(
            "/ocr/process",
            files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
            data={"output_format": "json"},
        )
        job_id = r.json()["job_id"]

        got = client.get(f"/ocr/result/{job_id}")
        assert got.status_code == 200
        assert got.json()["job_id"] == job_id
        assert got.json()["status"] == "complete"

    def test_unknown_job_404(self, client):
        assert client.get("/ocr/result/no-such-job").status_code == 404

    def test_job_list_after_process(self, client):
        client.post(
            "/ocr/process",
            files={"file": ("b.pdf", b"%PDF-1.4", "application/pdf")},
            data={"output_format": "json"},
        )
        r = client.get("/ocr/jobs?limit=5")
        assert r.status_code == 200
        assert any(j["status"] == "complete" for j in r.json()["jobs"])

    def test_searchable_pdf_404_when_not_generated(self, client):
        r = client.post(
            "/ocr/process",
            files={"file": ("c.pdf", b"%PDF-1.4", "application/pdf")},
            data={"output_format": "json"},  # no PDF requested
        )
        job_id = r.json()["job_id"]
        # No searchable PDF was generated, so download must 404
        assert client.get(f"/ocr/result/{job_id}/searchable.pdf").status_code == 404


# ---------------------------------------------------------------------------
# Audit trail (Stage 6)
# ---------------------------------------------------------------------------

class TestAudit:
    def test_full_pipeline_emits_all_stage_events(self, client):
        r = client.post(
            "/ocr/process",
            files={"file": ("l.pdf", b"%PDF-1.4", "application/pdf")},
            data={"output_format": "json"},
        )
        job_id = r.json()["job_id"]

        audit = client.get(f"/ocr/audit/{job_id}").json()
        stages = {e["stage"] for e in audit["events"] if e["stage"]}
        assert {"ingestion", "layout", "ocr", "postprocessing", "output"}.issubset(stages)

        types_ = [e["event_type"] for e in audit["events"]]
        assert types_[0] == "job_created"
        assert "result_stored" in types_

    def test_review_decision_recorded(self, client):
        r = client.post(
            "/ocr/process",
            files={"file": ("l.pdf", b"%PDF-1.4", "application/pdf")},
            data={"output_format": "json"},
        )
        job_id = r.json()["job_id"]

        rev = client.post(
            f"/ocr/review/{job_id}",
            data={
                "region_id": "2",
                "decision": "accept",
                "reviewer": "qc_test_user",
                "notes": "Verified Arabic dosage block",
            },
        )
        assert rev.status_code == 200

        audit = client.get(f"/ocr/audit/{job_id}").json()
        review_events = [e for e in audit["events"] if e["event_type"] == "review_decision"]
        assert len(review_events) == 1
        assert review_events[0]["actor"] == "qc_test_user"
        assert review_events[0]["payload"]["region_id"] == 2
        assert review_events[0]["payload"]["decision"] == "accept"

    def test_invalid_review_decision_rejected(self, client):
        r = client.post(
            "/ocr/process",
            files={"file": ("l.pdf", b"%PDF-1.4", "application/pdf")},
            data={"output_format": "json"},
        )
        job_id = r.json()["job_id"]

        rev = client.post(
            f"/ocr/review/{job_id}",
            data={
                "region_id": "2",
                "decision": "maybe",  # invalid
                "reviewer": "qc_test_user",
            },
        )
        assert rev.status_code == 400

    def test_review_for_unknown_job_404(self, client):
        rev = client.post(
            "/ocr/review/nope",
            data={"region_id": "1", "decision": "accept", "reviewer": "x"},
        )
        assert rev.status_code == 404
