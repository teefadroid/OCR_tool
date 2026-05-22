# PharmOCR — Mixed Arabic/English Pharmaceutical Leaflet OCR Pipeline

A production-grade OCR pipeline for extracting structured text from pharmaceutical package inserts containing mixed Arabic (RTL) and English (LTR) content.

## Architecture

```
PDF/Image Input
      │
      ▼
┌─────────────────────────────┐
│  Stage 1: Ingestion         │  pdf2image + OpenCV (300 DPI, deskew, denoise)
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 2: Layout Analysis   │  PP-DocLayoutV3 via GLM-OCR
└─────────────┬───────────────┘
              ▼
     ┌────────┴────────┐
     ▼                 ▼
┌──────────┐     ┌────────────────┐
│ GLM-OCR  │     │Arabic-GLM-OCR  │   Parallel dual-model OCR
│  (EN/LA) │     │    -v2 (AR)    │
└────┬─────┘     └───────┬────────┘
     └────────┬──────────┘
              ▼
┌─────────────────────────────┐
│  Stage 4: Post-Processing   │  BiDi fix, numeral conversion, INN matching
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 5: Output Generation │  JSON, Markdown, Searchable PDF
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 6: QC Review API     │  FastAPI + confidence flagging
└─────────────────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull models via Ollama
ollama pull glm-ocr
# Arabic model loaded via HuggingFace + custom Modelfile (see docs/setup.md)
ollama create arabic-glm-ocr -f ollama/Modelfile.arabic-glm-ocr

# 3. Run on a single leaflet
python -m pharma_ocr.cli process --input leaflet.pdf --output ./output

# 4. Generate searchable PDF + JSON + Markdown in one go
python -m pharma_ocr.cli process --input leaflet.pdf --output ./output --format all

# 5. Start the API server
uvicorn pharma_ocr.api.app:app --host 0.0.0.0 --port 8000

# 6. Run the test suite
pytest tests/ -v
```

## Project Structure

```
pharma_ocr/
├── ingestion/        # PDF → image, pre-processing
├── layout/           # PP-DocLayout region detection
├── ocr/              # Dual-model routing (EN + AR)
├── postprocessing/   # BiDi, numerals, INN matching
├── output/           # JSON / Markdown / Searchable PDF export
├── storage/          # SQLite audit log + persistent job store (Stage 6)
├── api/              # FastAPI service
└── cli.py            # Command-line interface
ollama/               # Modelfile templates for ollama create
tests/                # Unit tests for deterministic post-processing logic
data/
├── inn_list/         # WHO INN reference data
└── samples/          # Test leaflets
docs/                 # Setup and usage guides
```

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/ocr/health`                          | Both models reachable via Ollama? |
| `POST` | `/ocr/process`                         | Upload a PDF/image, run full pipeline |
| `GET`  | `/ocr/result/{job_id}`                 | Fetch a stored result |
| `GET`  | `/ocr/result/{job_id}/searchable.pdf`  | Download the searchable PDF for a job |
| `GET`  | `/ocr/jobs?limit=50`                   | List recent jobs |
| `GET`  | `/ocr/audit/{job_id}`                  | Full audit trail (regulatory compliance) |
| `POST` | `/ocr/review/{job_id}`                 | Record a QC reviewer decision |

## Configuration

The API server is configured via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PHARMOCR_OLLAMA_URL`     | `http://localhost:11434` | Ollama endpoint |
| `PHARMOCR_OUTPUT_DIR`     | `./api_output`           | Where exports are written |
| `PHARMOCR_AUDIT_DB`       | `./pharma_ocr_audit.sqlite3` | Audit log SQLite path |
| `PHARMOCR_CORS_ORIGINS`   | `http://localhost:3000,http://localhost:3001` | Allowed origins |

For production with Neon Postgres, swap the `AuditStore` connection layer
for `psycopg`/`asyncpg`. The schema in `pharma_ocr/storage/audit.py` is
already Postgres-compatible.

## Models Used

| Model | Purpose | Source |
|---|---|---|
| `zai-org/GLM-OCR` | English / Latin OCR | [GitHub](https://github.com/zai-org/GLM-OCR) |
| `sherif1313/Arabic-GLM-OCR-v2` | Arabic OCR | [HuggingFace](https://huggingface.co/sherif1313/Arabic-GLM-OCR-v1) |
| `PP-DocLayoutV3` | Layout analysis | Bundled in GLM-OCR |

## Requirements

- Python 3.10+
- Ollama (for local model serving)
- Node.js 18+ (for frontend)
- GPU recommended (RTX 3080+ for real-time processing)
- 8GB+ VRAM

## License

MIT License — See LICENSE file.
