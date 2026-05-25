"""Tests for the SQLite audit log + job store."""

from __future__ import annotations

from pathlib import Path

import pytest

from pharma_ocr.storage.audit import AuditEvent, AuditStore


@pytest.fixture
def store(tmp_path: Path) -> AuditStore:
    return AuditStore(db_path=tmp_path / "audit.sqlite3")


class TestJobLifecycle:
    def test_create_job(self, store: AuditStore):
        job_id = store.create_job(source_file="leaflet.pdf")
        assert job_id
        job = store.get_job(job_id)
        assert job is not None
        assert job["source_file"] == "leaflet.pdf"
        assert job["status"] == "queued"

    def test_create_job_with_explicit_id(self, store: AuditStore):
        job_id = store.create_job(source_file="x.pdf", job_id="abc12345")
        assert job_id == "abc12345"

    def test_update_status(self, store: AuditStore):
        job_id = store.create_job(source_file="x.pdf")
        store.update_status(job_id, "running", "started")
        job = store.get_job(job_id)
        assert job["status"] == "running"

    def test_store_result(self, store: AuditStore):
        job_id = store.create_job(source_file="x.pdf")
        result = {
            "pages_processed": 3,
            "inn_matches": [{"matched": "paracetamol"}],
            "drug_info": {"all_strengths": [{"value": "500", "unit": "mg"}]},
            "ocr_metadata": {"regions_flagged_for_review": [1, 2]},
            "text_content": {"combined": "Hello"},
        }
        store.store_result(job_id, result)
        job = store.get_job(job_id)
        assert job["status"] == "complete"
        assert job["pages"] == 3
        assert job["flagged"] == 2
        assert job["inn_count"] == 1
        assert job["strength_count"] == 1
        assert job["result"]["pages_processed"] == 3

    def test_get_nonexistent_job(self, store: AuditStore):
        assert store.get_job("nope") is None

    def test_list_jobs_orders_by_created_desc(self, store: AuditStore):
        a = store.create_job(source_file="a.pdf")
        b = store.create_job(source_file="b.pdf")
        c = store.create_job(source_file="c.pdf")
        jobs = store.list_jobs()
        ids = [j["job_id"] for j in jobs]
        assert ids == [c, b, a]


class TestAuditTrail:
    def test_events_recorded_in_order(self, store: AuditStore):
        job_id = store.create_job(source_file="x.pdf")
        store.log_event(AuditEvent(
            job_id=job_id, event_type="stage_complete", stage="ingestion",
            message="rendered",
        ))
        store.log_event(AuditEvent(
            job_id=job_id, event_type="stage_complete", stage="ocr",
            message="ocr done",
        ))
        events = store.get_audit_trail(job_id)
        # First event is the auto-created job_created event from create_job
        assert events[0]["event_type"] == "job_created"
        assert events[1]["stage"] == "ingestion"
        assert events[2]["stage"] == "ocr"

    def test_payload_round_trips(self, store: AuditStore):
        job_id = store.create_job(source_file="x.pdf")
        store.log_event(AuditEvent(
            job_id=job_id, event_type="stage_complete", stage="ocr",
            payload={"regions": 5, "model": "glm-ocr"},
        ))
        events = store.get_audit_trail(job_id)
        ocr_event = next(e for e in events if e["stage"] == "ocr")
        assert ocr_event["payload"] == {"regions": 5, "model": "glm-ocr"}

    def test_review_decision_recorded(self, store: AuditStore):
        job_id = store.create_job(source_file="x.pdf")
        store.record_review_decision(
            job_id=job_id, region_id=3, decision="accept",
            reviewer="qc_user_1", notes="Looks fine",
        )
        events = store.get_audit_trail(job_id)
        review = next(e for e in events if e["event_type"] == "review_decision")
        assert review["actor"] == "qc_user_1"
        assert review["payload"]["region_id"] == 3
        assert review["payload"]["decision"] == "accept"


class TestPersistence:
    def test_data_persists_across_instances(self, tmp_path: Path):
        db_path = tmp_path / "audit.sqlite3"

        store1 = AuditStore(db_path=db_path)
        job_id = store1.create_job(source_file="leaflet.pdf")
        store1.update_status(job_id, "running")

        # New instance, same DB
        store2 = AuditStore(db_path=db_path)
        job = store2.get_job(job_id)
        assert job is not None
        assert job["status"] == "running"

    def test_failure_status_recorded_as_error_event(self, store: AuditStore):
        job_id = store.create_job(source_file="x.pdf")
        store.update_status(job_id, "failed", "boom")
        events = store.get_audit_trail(job_id)
        error = next(e for e in events if e["event_type"] == "error")
        assert "boom" in error["message"]
