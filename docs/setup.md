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
Download from https://github.com/oschwartz10612/poppler-windows/releases and add to PATH.

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 3. Pull OCR models via Ollama

```bash
# Start Ollama
ollama serve

# Pull GLM-OCR base (English/Latin)
ollama pull glm-ocr
```

### Arabic-GLM-OCR-v2 (Hugging Face)

The Arabic model is not yet in the Ollama registry. Load it manually:

```bash
# Option A: Download from Hugging Face and convert to GGUF
pip install huggingface_hub
huggingface-cli download sherif1313/Arabic-GLM-OCR-v2 --local-dir ./models/arabic-glm-ocr

# Option B: Use the HuggingFace Transformers API directly
# Set ar_model="hf:sherif1313/Arabic-GLM-OCR-v2" in OCRRouter config
# (see pharma_ocr/ocr/arabic_client.py for HF fallback mode)
```

## 4. Add WHO INN List

Download the current WHO INN list and place it at:
```
data/inn_list/who_inn.csv
```

Format: one INN per line in column 1.

## 5. Run the API server

```bash
uvicorn pharma_ocr.api.app:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

## 6. CLI usage

```bash
# Process a single file
python -m pharma_ocr.cli process --input leaflet.pdf --output ./output

# Batch process a folder
python -m pharma_ocr.cli batch --input-dir ./leaflets --output ./output

# Check model health
python -m pharma_ocr.cli health
```

## 7. Next.js frontend integration

The API exposes CORS-enabled endpoints for localhost:3000 by default.
Update `allow_origins` in `pharma_ocr/api/app.py` for your deployment URL.

```typescript
// Example Next.js API call
const formData = new FormData();
formData.append('file', file);
formData.append('output_format', 'both');

const res = await fetch('http://localhost:8000/ocr/process', {
  method: 'POST',
  body: formData,
});
const result = await res.json();
```
