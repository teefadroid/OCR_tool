"""
PharmOCR FastAPI Service

Exposes the full OCR pipeline as an HTTP API for integration with
the Next.js frontend and external systems.

Endpoints:
  POST /ocr/process          — Upload a PDF or image; returns JSON extraction
  GET  /ocr/health           — Check model availability (GLM-OCR + Arabic)
  GET  /ocr/result/{job_id}  — Retrieve a previous result by job ID
  GET  /ocr/jobs             — List recent jobs
  GET  /ocr/audit/{job_id}   — Full audit trail for a job (Stage 6)
  POST /ocr/review/{job_id}  — Record a QC reviewer decision

All requests are processed locally — no data leaves the server.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from ..ingestion.preprocessor import DocumentPreprocessor
from ..layout.analyzer import LayoutAnalyzer
from ..ocr.router import OCRRouter
from ..output.json_exporter import JSONExporter
from ..output.markdown_exporter import MarkdownExporter
from ..output.pdf_overlay import PDFOverlay
from ..postprocessing.pipeline import PostProcessor
from ..storage.audit import AuditEvent, AuditStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PharmOCR API",
    description="Mixed Arabic/English pharmaceutical leaflet OCR pipeline",
    version="1.0.0",
)

_ALLOWED_ORIGINS = os.getenv(
    "PHARMOCR_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared service instances (stateless; reused across requests)
_OUTPUT_DIR = Path(os.getenv("PHARMOCR_OUTPUT_DIR", "./api_output"))
_DB_PATH = os.getenv("PHARMOCR_AUDIT_DB", "./pharma_ocr_audit.sqlite3")
_OLLAMA_URL = os.getenv("PHARMOCR_OLLAMA_URL", "http://localhost:11434")

_preprocessor = DocumentPreprocessor(dpi=300)
_layout = LayoutAnalyzer(mode="ollama", ollama_url=_OLLAMA_URL)
_router = OCRRouter(ollama_url=_OLLAMA_URL)
_json_exporter = JSONExporter(output_dir=_OUTPUT_DIR)
_md_exporter = MarkdownExporter(output_dir=_OUTPUT_DIR)
_pdf_overlay = PDFOverlay(output_dir=_OUTPUT_DIR)
_audit = AuditStore(db_path=_DB_PATH)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/ocr/health", tags=["System"])
def health_check():
    """Check that both OCR models are reachable via Ollama."""
    models = _router.health_check()
    all_ok = all(models.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "ok" if all_ok else "degraded",
            "models": models,
            "message": "All models available" if all_ok
                        else "One or more models unavailable — check Ollama",
        }
    )


# ---------------------------------------------------------------------------
# Single file processing
# ---------------------------------------------------------------------------

@app.post("/ocr/process", tags=["OCR"])
async def process_document(
    file: UploadFile = File(..., description="PDF or image file"),
    output_format: str = Form(default="json", description="json | markdown | pdf | all"),
    confidence_threshold: float = Form(default=0.85),
):
    """
    Upload a pharmaceutical leaflet and receive structured OCR output.

    Supports: PDF, JPG, PNG, TIFF, BMP, WEBP
    Returns: JSON with extracted text, drug names, strengths, INN matches.
    Side effects: creates a job row in the audit store and records each stage.
    """
    allowed = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    suffix = Path(file.filename or "file").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    job_id = _audit.create_job(source_file=file.filename or "unknown")

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        logger.info("[%s] Processing: %s", job_id, file.filename)
        _audit.update_status(job_id, "running", "Pipeline started")

        # Stage 1: Ingestion
        pages = _preprocessor.process(tmp_path)
        _audit.log_event(AuditEvent(
            job_id=job_id, event_type="stage_complete", stage="ingestion",
            message=f"Rendered {len(pages)} page(s) at 300 DPI",
        ))

        # Stage 2: Layout
        all_regions = []
        for page in pages:
            all_regions.extend(_layout.analyze(page))
        _audit.log_event(AuditEvent(
            job_id=job_id, event_type="stage_complete", stage="layout",
            message=f"Detected {len(all_regions)} region(s)",
        ))

        # Stage 3: Dual-model OCR
        all_regions = _router.process_regions(all_regions)
        _audit.log_event(AuditEvent(
            job_id=job_id, event_type="stage_complete", stage="ocr",
            message="Dual-model OCR complete (en + ar)",
        ))

        # Stage 4: Post-processing (per-request to honor confidence threshold)
        postprocessor = PostProcessor(confidence_threshold=confidence_threshold)
        result = postprocessor.process(all_regions, source_file=file.filename or "")
        _audit.log_event(AuditEvent(
            job_id=job_id, event_type="stage_complete", stage="postprocessing",
            message=(
                f"{len(result.inn_matches)} INN matches, "
                f"{len(result.strengths)} strengths, "
                f"{len(result.regions_flagged_for_review)} flagged"
            ),
        ))

        # Stage 5: Export
        result_dict = _json_exporter.to_dict(result)
        result_dict["job_id"] = job_id

        if output_format in ("markdown", "all"):
            result_dict["markdown"] = _md_exporter.to_string(result)

        if output_format in ("pdf", "all") and suffix == ".pdf":
            regions_per_page: dict[int, list] = {}
            for r in all_regions:
                page_num = r.metadata.get("page", 1)
                regions_per_page.setdefault(page_num, []).append(r)
            pdf_path = _pdf_overlay.create_searchable_pdf(
                tmp_path,
                regions_per_page,
                filename=f"{job_id}_searchable.pdf",
            )
            if pdf_path:
                result_dict["searchable_pdf"] = str(pdf_path)

        # Persist
        _audit.store_result(job_id, result_dict)
        return JSONResponse(content=result_dict)

    except Exception as exc:
        logger.error("[%s] Pipeline error: %s", job_id, exc, exc_info=True)
        _audit.update_status(job_id, "failed", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Result retrieval
# ---------------------------------------------------------------------------

@app.get("/ocr/result/{job_id}", tags=["OCR"])
def get_result(job_id: str):
    """Retrieve a previous OCR job result by job ID."""
    job = _audit.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JSONResponse(content=job)


@app.get("/ocr/result/{job_id}/searchable.pdf", tags=["OCR"])
def get_searchable_pdf(job_id: str):
    """Download the searchable PDF for a completed job, if generated."""
    pdf_file = _OUTPUT_DIR / f"{job_id}_searchable.pdf"
    if not pdf_file.exists():
        raise HTTPException(status_code=404, detail="Searchable PDF not found for this job")
    return FileResponse(pdf_file, media_type="application/pdf", filename=pdf_file.name)


@app.get("/ocr/jobs", tags=["OCR"])
def list_jobs(limit: int = 50):
    """List recent OCR jobs (latest first)."""
    return {"jobs": _audit.list_jobs(limit=limit)}


# ---------------------------------------------------------------------------
# Audit trail (Stage 6)
# ---------------------------------------------------------------------------

@app.get("/ocr/audit/{job_id}", tags=["QC"])
def get_audit_trail(job_id: str):
    """Return the full audit trail for a job — required for SFDA/MoH submissions."""
    job = _audit.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "events": _audit.get_audit_trail(job_id),
    }


@app.post("/ocr/review/{job_id}", tags=["QC"])
def record_review(
    job_id: str,
    region_id: int = Form(...),
    decision: str = Form(..., description="accept | reject | edit"),
    reviewer: str = Form(...),
    notes: str = Form(default=""),
):
    """Record a QC reviewer's decision on a flagged region."""
    job = _audit.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if decision not in {"accept", "reject", "edit"}:
        raise HTTPException(status_code=400, detail="decision must be accept | reject | edit")
    _audit.record_review_decision(
        job_id=job_id,
        region_id=region_id,
        decision=decision,
        reviewer=reviewer,
        notes=notes,
    )
    return {"status": "recorded", "job_id": job_id, "region_id": region_id}
