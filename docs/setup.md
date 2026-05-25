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

### Arabic-GLM-OCR-v2 — known Ollama limitation

The Arabic fine-tune is **not** in the Ollama registry, and Ollama's
Modelfile does not currently support attaching a vision projector
(`mmproj`) to a custom GGUF
([ollama/ollama#9967](https://github.com/ollama/ollama/issues/9967)).
That means even if you import the community-converted GGUF
([Makadi86/Arabic-GLM-OCR-v2-GGUF](https://huggingface.co/Makadi86/Arabic-GLM-OCR-v2-GGUF)),
the resulting Ollama model has no vision capability and OCR will fail.

If you saw this error:

```
Error: 400 Bad Request: invalid model name
```

it means the `FROM` path in the Modelfile resolved to a missing file and
Ollama tried to interpret the path as a registry name. Even after fixing
the path the resulting model will not work for OCR (see the limitation
above).

Three working options:

#### Option 1 — single-model fallback (recommended for first-run validation)

Use the registry `glm-ocr` model for both languages. The base GLM-OCR
has some Arabic capability — lower accuracy than the fine-tune but
sufficient to validate the rest of the pipeline today, with zero extra
setup.

```bash
ollama pull glm-ocr

# CLI
python -m pharma_ocr.cli process --input leaflet.pdf \
    --en-model glm-ocr --ar-model glm-ocr

# API server (Linux/macOS)
export PHARMOCR_AR_MODEL=glm-ocr
uvicorn pharma_ocr.api.app:app --port 8000

# API server (Windows PowerShell)
$env:PHARMOCR_AR_MODEL = "glm-ocr"
uvicorn pharma_ocr.api.app:app --port 8000
```

The CLI will print a yellow "running in single-model mode" notice so
you know which configuration is in effect.

#### Option 2 — HuggingFace transformers (high-accuracy Arabic)

Run the Arabic model in-process via `transformers`, bypassing Ollama
entirely for that language. Adds ~3 GB of dependencies but gets you
full Arabic OCR accuracy:

```bash
pip install torch torchvision transformers accelerate
huggingface-cli download sherif1313/Arabic-GLM-OCR-v2 \
    --local-dir ./models/arabic-glm-ocr

# Future flag — implementation pending; tracked as a follow-up:
# export PHARMOCR_AR_BACKEND=hf
```

> Status: requires a small `arabic_hf_client.py` adapter in
> `pharma_ocr/ocr/`. Not in the current release; let us know if you
> want it prioritised.

#### Option 3 — llama-server (llama.cpp) for the Arabic model

`llama.cpp` natively supports `--mmproj`. Run the Arabic model on a
separate port and point PharmOCR at it:

```bash
# Download both files (1.35 GB + ~600 MB)
huggingface-cli download Makadi86/Arabic-GLM-OCR-v2-GGUF \
    --local-dir ./models/arabic-glm-ocr-gguf

# Build llama.cpp (or grab a release binary)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j

# Serve on port 8080 with vision
./build/bin/llama-server \
    -m ../models/arabic-glm-ocr-gguf/arabic-glm-ocr-v2-fixed.gguf \
    --mmproj ../models/arabic-glm-ocr-gguf/mmproj-arabic-glm-ocr-v2.gguf \
    --port 8080
```

> Status: requires a small `arabic_llamacpp_client.py` adapter; tracked
> as a follow-up in the same way as Option 2.

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
| `PHARMOCR_EN_MODEL`     | `glm-ocr`                       | Latin/English OCR model name in Ollama |
| `PHARMOCR_AR_MODEL`     | `arabic-glm-ocr`                | Arabic OCR model name. Set to `glm-ocr` for single-model fallback (see "Arabic-GLM-OCR-v2 — known Ollama limitation" below). |

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
