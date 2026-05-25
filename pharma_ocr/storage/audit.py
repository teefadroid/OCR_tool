"""
Audit Log & Persistent Job Store (Stage 6 of the architecture plan).

The architecture plan calls for:
  > extraction results, confidence scores per section, and the full audit
  > trail are stored in your Neon PostgreSQL database — critical for
  > regulatory compliance documentation in SFDA/MoH submissions.

This module provides a backend-agnostic interface implemented today against
SQLite (zero-dependency, ideal for local + CI). The same schema is
Postgres-compatible: switching to Neon means swapping the connection
string and using a Postgres driver — no schema changes required.

Two entities are recorded:
  - jobs       — one row per OCR job (file processed, status, summary)
  - audit_log  — append-only event log (start, stage_complete, error,
                 review_decision) for full regulatory traceability.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """A single event in the audit trail."""
    job_id: str
    event_type: str            # job_created | stage_complete | error | review_decision | result_stored
    stage: str = ""            # ingestion | layout | ocr | postprocessing | output | qc
    message: str = ""
    actor: str = "system"      # 'system' or a user identifier
    payload: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    source_file   TEXT NOT NULL,
    status        TEXT NOT NULL,
    pages         INTEGER DEFAULT 0,
    flagged       INTEGER DEFAULT 0,
    inn_count     INTEGER DEFAULT 0,
    strength_count INTEGER DEFAULT 0,
    result_json   TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        TEXT NOT NULL,
    event_type    TEXT NOT NULL,
    stage         TEXT,
    message       TEXT,
    actor         TEXT,
    payload_json  TEXT,
    timestamp     TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE INDEX IF NOT EXISTS idx_audit_job_id ON audit_log(job_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
"""


class AuditStore:
    """
    Thread-safe SQLite-backed audit log + job store.

    Production note: swap the connection layer to psycopg/asyncpg for Neon.
    The schema (`_SCHEMA_SQL`) is plain ANSI SQL apart from
    `INTEGER PRIMARY KEY AUTOINCREMENT` which becomes `BIGSERIAL` in Postgres.
    """

    def __init__(self, db_path: str | Path = "./pharma_ocr_audit.sqlite3"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._conn() as conn:
            for stmt in _SCHEMA_SQL.split(";"):
                if stmt.strip():
                    conn.execute(stmt)

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def create_job(self, source_file: str, job_id: Optional[str] = None) -> str:
        """Insert a new job in 'queued' status. Returns the job_id."""
        job_id = job_id or str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, source_file, status, created_at, updated_at) "
                "VALUES (?, ?, 'queued', ?, ?)",
                (job_id, source_file, now, now),
            )
        self.log_event(AuditEvent(
            job_id=job_id,
            event_type="job_created",
            message=f"Job created for {source_file}",
        ))
        return job_id

    def update_status(self, job_id: str, status: str, message: str = "") -> None:
        """Update a job's status (running, complete, failed)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (status, now, job_id),
            )
        self.log_event(AuditEvent(
            job_id=job_id,
            event_type="stage_complete" if status != "failed" else "error",
            message=message or status,
        ))

    def store_result(self, job_id: str, result: dict) -> None:
        """Persist the final OCR result dict for a job."""
        now = datetime.now(timezone.utc).isoformat()
        pages = int(result.get("pages_processed", 0))
        flagged = len(result.get("ocr_metadata", {}).get("regions_flagged_for_review", []))
        inn_count = len(result.get("inn_matches", []))
        strength_count = len(result.get("drug_info", {}).get("all_strengths", []))
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status='complete', result_json=?, pages=?, flagged=?, "
                "inn_count=?, strength_count=?, updated_at=? WHERE job_id=?",
                (json.dumps(result, ensure_ascii=False), pages, flagged,
                 inn_count, strength_count, now, job_id),
            )
        self.log_event(AuditEvent(
            job_id=job_id,
            event_type="result_stored",
            stage="output",
            message=f"Stored result: {pages} pages, {flagged} flagged, {inn_count} INN matches",
            payload={"pages": pages, "flagged": flagged, "inn_count": inn_count},
        ))

    def get_job(self, job_id: str) -> Optional[dict]:
        """Return a job row + parsed result, or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("result_json"):
            try:
                d["result"] = json.loads(d["result_json"])
            except json.JSONDecodeError:
                d["result"] = None
        d.pop("result_json", None)
        return d

    def list_jobs(self, limit: int = 50) -> list[dict]:
        """Return the most recent jobs (latest first), without full result blob."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT job_id, source_file, status, pages, flagged, inn_count, "
                "strength_count, created_at, updated_at FROM jobs "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def log_event(self, event: AuditEvent) -> None:
        """Append a single audit event."""
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO audit_log (job_id, event_type, stage, message, actor, "
                "payload_json, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    event.job_id, event.event_type, event.stage, event.message,
                    event.actor,
                    json.dumps(event.payload, ensure_ascii=False) if event.payload else None,
                    event.timestamp,
                ),
            )

    def log_events(self, events: Iterable[AuditEvent]) -> None:
        for e in events:
            self.log_event(e)

    def get_audit_trail(self, job_id: str) -> list[dict]:
        """Return all audit events for a job in chronological order."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, job_id, event_type, stage, message, actor, "
                "payload_json, timestamp FROM audit_log WHERE job_id = ? "
                "ORDER BY id ASC",
                (job_id,),
            ).fetchall()
        events = []
        for r in rows:
            d = dict(r)
            if d.get("payload_json"):
                try:
                    d["payload"] = json.loads(d["payload_json"])
                except json.JSONDecodeError:
                    d["payload"] = None
            d.pop("payload_json", None)
            events.append(d)
        return events

    def record_review_decision(
        self,
        job_id: str,
        region_id: int,
        decision: str,
        reviewer: str,
        notes: str = "",
    ) -> None:
        """Record a human QC decision on a flagged region."""
        self.log_event(AuditEvent(
            job_id=job_id,
            event_type="review_decision",
            stage="qc",
            actor=reviewer,
            message=f"Region {region_id}: {decision}",
            payload={"region_id": region_id, "decision": decision, "notes": notes},
        ))
