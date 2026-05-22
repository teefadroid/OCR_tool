"""
PharmOCR FastAPI Service

Exposes the full OCR pipeline as an HTTP API for integration with
the Next.js frontend and external systems.

Endpoints:
  POST /ocr/process         — Upload a PDF or image; returns JSON extraction
  GET  /ocr/health          — Check model availability (GLM-OCR + Arabic)
  GET  /ocr/result/{job_id} — Retrieve a previous result by job ID
  POST /ocr/batch           — Queue multiple files for batch processing

All requests are processed locally — no data leaves the server.
"""

from __future__ import annotations

import logging
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
from ..postprocessing.pipeline import PostProcessor
from ..output.json_exporter import JSONExporter
from ..output.markdown_exporter import MarkdownExporter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PharmOCR API",
    description="Mixed Arabic/English pharmaceutical leaflet OCR pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared service instances
_preprocessor = DocumentPreprocessor(dpi=300)
_layout = LayoutAnalyzer(mode="ollama")
_router = OCRRouter()
_postprocessor = PostProcessor()
_json_exporter = JSONExporter(output_dir="./api_output")
_md_exporter = MarkdownExporter(output_dir="./api_output")

# In-memory job store (replace with DB for production)
_job_store: dict[str, dict] = {}


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
    output_format: str = Form(default="json", description="json | markdown | both"),
    confidence_threshold: float = Form(default=0.85),
):
    """
    Upload a pharmaceutical leaflet and receive structured OCR output.

    Supports: PDF, JPG, PNG, TIFF, BMP, WEBP
    Returns: JSON with extracted text, drug names, strengths, INN matches
    """
    job_id = str(uuid.uuid4())[:8]
    allowed = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

    suffix = Path(file.filename or "file").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        logger.info("[%s] Processing: %s", job_id, file.filename)

        # Stage 1: Ingestion
        pages = _preprocessor.process(tmp_path)

        # Stage 2: Layout
        all_regions = []
        for page in pages:
            regions = _layout.analyze(page)
            all_regions.extend(regions)

        # Stage 3: Dual-model OCR
        all_regions = _router.process_regions(all_regions)

        # Stage 4: Post-processing
        postprocessor = PostProcessor(confidence_threshold=confidence_threshold)
        result = postprocessor.process(all_regions, source_file=file.filename or "")

        # Stage 5: Export
        result_dict = _json_exporter.to_dict(result)
        result_dict["job_id"] = job_id

        if output_format in ("markdown", "both"):
            md_text = _md_exporter.to_string(result)
            result_dict["markdown"] = md_text

        # Store for retrieval
        _job_store[job_id] = result_dict

        return JSONResponse(content=result_dict)

    except Exception as exc:
        logger.error("[%s] Pipeline error: %s", job_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Result retrieval
# ---------------------------------------------------------------------------

@app.get("/ocr/result/{job_id}", tags=["OCR"])
def get_result(job_id: str):
    """Retrieve a previous OCR job result by job ID."""
    result = _job_store.get(job_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# List recent jobs
# ---------------------------------------------------------------------------

@app.get("/ocr/jobs", tags=["OCR"])
def list_jobs(limit: int = 20):
    """List the most recent OCR jobs (latest first)."""
    jobs = [
        {"job_id": k, "source_file": v.get("source_file"), "pages": v.get("pages_processed")}
        for k, v in list(_job_store.items())[-limit:]
    ]
    return {"jobs": list(reversed(jobs)), "total": len(_job_store)}
