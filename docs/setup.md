# PharmOCR Setup Guide

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Poppler (required by pdf2image)
- 8 GB+ VRAM recommended (RTX 3080 or better)

## 1. Install Poppler

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
Download from <https://github.com/oschwartz10612/poppler-windows/releases> and add to `PATH`.

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 3. Pull OCR models via Ollama

```bash
# Start Ollama
ollama serve

# Pull GLM-OCR base (English / Latin)
ollama pull glm-ocr
```

Optional: build a pharma-tuned variant from the bundled Modelfile so the
system prompt biases the model toward leaflet structure (tables,
headings, dosage):

```bash
ollama create glm-ocr-pharma -f ollama/Modelfile.glm-ocr
# then run with: --en-model glm-ocr-pharma
```

### Arabic-GLM-OCR-v2 (Hugging Face)

The Arabic model is not yet in the Ollama registry. Build it locally:

```bash
# 1. Download model from HuggingFace
pip install huggingface_hub
huggingface-cli download sherif1313/Arabic-GLM-OCR-v2 \
    --local-dir ./models/arabic-glm-ocr

# 2. (If only safetensors are published) convert to GGUF using llama.cpp:
#    python convert.py ./models/arabic-glm-ocr --outtype q4_k_m

# 3. Build the Ollama model from the bundled Modelfile
ollama create arabic-glm-ocr -f ollama/Modelfile.arabic-glm-ocr

# 4. Verify
ollama run arabic-glm-ocr "صورة اختبار"
```

The bundled `ollama/Modelfile.arabic-glm-ocr` already contains an Arabic
system prompt and OCR-tuned sampling parameters (`temperature=0`,
`top_k=1`).

## 4. Add the WHO INN List

Download the current WHO INN list and place it at:

```
data/inn_list/who_inn.csv
```

Format: one INN per line in column 1 (other columns ignored).
Without this file, PharmOCR falls back to a built-in list of ~40 common
INNs — enough for development but not for regulatory submissions.

## 5. Run the API server

```bash
uvicorn pharma_ocr.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs: <http://localhost:8000/docs>

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PHARMOCR_OLLAMA_URL`   | `http://localhost:11434`        | Ollama HTTP endpoint |
| `PHARMOCR_OUTPUT_DIR`   | `./api_output`                  | Where exports are written |
| `PHARMOCR_AUDIT_DB`     | `./pharma_ocr_audit.sqlite3`    | Audit DB file (SQLite) |
| `PHARMOCR_CORS_ORIGINS` | `http://localhost:3000,http://localhost:3001` | CSV of allowed origins |

## 6. CLI usage

```bash
# Process a single file (JSON only)
python -m pharma_ocr.cli process --input leaflet.pdf --output ./output

# JSON + Markdown
python -m pharma_ocr.cli process --input leaflet.pdf --format both

# JSON + Markdown + Searchable PDF (regulatory archive grade)
python -m pharma_ocr.cli process --input leaflet.pdf --format all

# Searchable PDF only
python -m pharma_ocr.cli process --input leaflet.pdf --format pdf

# Batch process a folder
python -m pharma_ocr.cli batch --input-dir ./leaflets --output ./output

# Check model health
python -m pharma_ocr.cli health
```

## 7. Audit log & QC review (Stage 6)

Every API request is automatically logged to the audit store
(`PHARMOCR_AUDIT_DB`, default SQLite at `./pharma_ocr_audit.sqlite3`).
The schema records:

- One row per **job** (`source_file`, `status`, `pages`, `flagged`,
  `inn_count`, `strength_count`, full `result_json`).
- Append-only **audit_log** events: `job_created`, `stage_complete`
  per pipeline stage, `result_stored`, `error`, `review_decision`.

Retrieve the full trail for a job:

```bash
curl http://localhost:8000/ocr/audit/$JOB_ID
```

Record a QC reviewer decision on a flagged region:

```bash
curl -X POST http://localhost:8000/ocr/review/$JOB_ID \
     -F region_id=2 \
     -F decision=accept \
     -F reviewer="qc_user_1" \
     -F notes="Verified Arabic dosage block"
```

`decision` must be one of `accept | reject | edit`.

### Switching to Neon Postgres for production

The schema in `pharma_ocr/storage/audit.py` is plain ANSI SQL except
`INTEGER PRIMARY KEY AUTOINCREMENT` — replace that with `BIGSERIAL`
when you swap the connection layer to `psycopg`/`asyncpg`. No other
schema changes are required for the move from local SQLite to Neon.

## 8. Next.js frontend integration

The API exposes CORS-enabled endpoints. Update `PHARMOCR_CORS_ORIGINS`
for your deployment URL.

```typescript
// Example Next.js API call
const formData = new FormData();
formData.append('file', file);
formData.append('output_format', 'all'); // json + markdown + searchable PDF

const res = await fetch('http://localhost:8000/ocr/process', {
  method: 'POST',
  body: formData,
});
const result = await res.json();

// QC review screen — fetch audit trail
const audit = await fetch(`http://localhost:8000/ocr/audit/${result.job_id}`)
  .then(r => r.json());

// Submit a review decision
await fetch(`http://localhost:8000/ocr/review/${result.job_id}`, {
  method: 'POST',
  body: new URLSearchParams({
    region_id: '2',
    decision: 'accept',
    reviewer: 'qc_user_1',
    notes: 'Verified',
  }),
});
```

## 9. Running the test suite

```bash
pytest tests/ -v
```

The tests stub heavy native dependencies (numpy / cv2 / pdf2image), so
they run in any clean Python environment without GPU or model
installation. They cover the deterministic post-processing logic
(numerals, BiDi, INN matching) and the audit store.
