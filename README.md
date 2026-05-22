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
# Arabic model loaded via HuggingFace (see docs/setup.md)

# 3. Run on a single leaflet
python -m pharma_ocr.cli process --input leaflet.pdf --output ./output

# 4. Start the API server
uvicorn pharma_ocr.api.app:app --host 0.0.0.0 --port 8000

# 5. Open review UI (Next.js)
cd frontend && npm install && npm run dev
```

## Project Structure

```
pharma_ocr/
├── ingestion/        # PDF → image, pre-processing
├── layout/           # PP-DocLayout region detection
├── ocr/              # Dual-model routing (EN + AR)
├── postprocessing/   # BiDi, numerals, INN matching
├── output/           # JSON/Markdown/PDF export
├── api/              # FastAPI service
└── cli.py            # Command-line interface
frontend/             # Next.js review UI
data/
├── inn_list/         # WHO INN reference data
└── samples/          # Test leaflets
docs/                 # Setup and usage guides
```

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
