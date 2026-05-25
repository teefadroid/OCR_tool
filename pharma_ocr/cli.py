"""
PharmOCR Command-Line Interface

Usage:
    # Process a single leaflet
    python -m pharma_ocr.cli process --input leaflet.pdf --output ./output

    # Process with Markdown output as well
    python -m pharma_ocr.cli process --input leaflet.pdf --output ./output --format both

    # Health check (verify models are available)
    python -m pharma_ocr.cli health

    # Batch process a folder of PDFs
    python -m pharma_ocr.cli batch --input-dir ./leaflets --output ./output
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    _RICH = True
except ImportError:
    _RICH = False

from .ingestion.preprocessor import DocumentPreprocessor
from .layout.analyzer import LayoutAnalyzer
from .ocr.router import OCRRouter
from .postprocessing.pipeline import PostProcessor
from .output.json_exporter import JSONExporter
from .output.markdown_exporter import MarkdownExporter
from .output.pdf_overlay import PDFOverlay

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

if _RICH:
    app = typer.Typer(help="PharmOCR — Mixed Arabic/English Pharmaceutical Leaflet OCR")
    console = Console()


def _run_pipeline(
    input_path: Path,
    output_dir: Path,
    fmt: str = "json",
    ollama_url: str = "http://localhost:11434",
    dpi: int = 300,
    confidence: float = 0.85,
    en_model: str = "glm-ocr",
    ar_model: str = "arabic-glm-ocr",
) -> dict:
    """Core pipeline runner. Returns result dict.

    Supported formats: 'json' | 'markdown' | 'pdf' | 'both' (json+md) | 'all' (json+md+pdf)

    Model overrides:
        en_model: name of the Latin/English OCR model in Ollama (default 'glm-ocr')
        ar_model: name of the Arabic OCR model. Set to the same value as en_model
                  to fall back to single-model operation when the Arabic
                  fine-tune isn't available locally.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Ingestion
    preprocessor = DocumentPreprocessor(dpi=dpi)
    pages = preprocessor.process(input_path)

    # Stage 2: Layout
    layout = LayoutAnalyzer(mode="ollama", ollama_url=ollama_url)
    all_regions = []
    for page in pages:
        all_regions.extend(layout.analyze(page))

    # Stage 3: Dual-model OCR
    router = OCRRouter(ollama_url=ollama_url, en_model=en_model, ar_model=ar_model)
    all_regions = router.process_regions(all_regions)

    # Stage 4: Post-processing
    postprocessor = PostProcessor(confidence_threshold=confidence)
    result = postprocessor.process(all_regions, source_file=str(input_path))

    # Stage 5: Export
    json_exp = JSONExporter(output_dir)
    json_exp.export(result)

    if fmt in ("markdown", "both", "all"):
        md_exp = MarkdownExporter(output_dir)
        md_exp.export(result)

    if fmt in ("pdf", "all") and input_path.suffix.lower() == ".pdf":
        pdf_exp = PDFOverlay(output_dir)
        # Group regions by page number for the overlay stage
        regions_per_page: dict[int, list] = {}
        for r in all_regions:
            page_num = r.metadata.get("page", 1)
            regions_per_page.setdefault(page_num, []).append(r)
        pdf_exp.create_searchable_pdf(input_path, regions_per_page)

    return json_exp.to_dict(result)


if _RICH:
    @app.command()
    def process(
        input: Path = typer.Option(..., "--input", "-i", help="PDF or image file"),
        output: Path = typer.Option(Path("./output"), "--output", "-o", help="Output directory"),
        format: str = typer.Option("json", "--format", "-f", help="json | markdown | pdf | both | all"),
        ollama_url: str = typer.Option("http://localhost:11434", help="Ollama server URL"),
        dpi: int = typer.Option(300, help="DPI for PDF rendering"),
        confidence: float = typer.Option(0.85, help="Confidence threshold for QC flags"),
        en_model: str = typer.Option("glm-ocr", help="Latin/English OCR model name in Ollama"),
        ar_model: str = typer.Option(
            "arabic-glm-ocr",
            help=(
                "Arabic OCR model name in Ollama. "
                "Pass --ar-model glm-ocr to fall back to single-model mode "
                "if the Arabic fine-tune isn't available locally."
            ),
        ),
    ):
        """Process a single pharmaceutical leaflet PDF or image."""
        if not input.exists():
            console.print(f"[red]Error:[/red] File not found: {input}")
            raise typer.Exit(1)

        console.print(f"[bold green]PharmOCR[/bold green] processing: [cyan]{input}[/cyan]")
        if en_model == ar_model:
            console.print(
                f"[yellow]Note:[/yellow] running in single-model mode "
                f"(both languages → [cyan]{en_model}[/cyan])"
            )

        result = _run_pipeline(
            input, output, format, ollama_url, dpi, confidence,
            en_model=en_model, ar_model=ar_model,
        )

        # Summary table
        t = Table(title="Extraction Summary")
        t.add_column("Field", style="bold")
        t.add_column("Value")
        t.add_row("Source", str(input))
        t.add_row("Pages", str(result.get("pages_processed", 0)))
        t.add_row("INN Matches", str(len(result.get("inn_matches", []))))
        t.add_row(
            "Strengths Found",
            str(len(result.get("drug_info", {}).get("all_strengths", []))),
        )
        t.add_row(
            "Combined Text Chars",
            str(len(result.get("text_content", {}).get("combined", ""))),
        )
        t.add_row(
            "Regions for Review",
            str(len(result.get("ocr_metadata", {}).get("regions_flagged_for_review", []))),
        )
        console.print(t)
        console.print(f"[green]✓ Output written to:[/green] {output}")

    @app.command()
    def health(
        ollama_url: str = typer.Option("http://localhost:11434", help="Ollama server URL"),
        en_model: str = typer.Option("glm-ocr"),
        ar_model: str = typer.Option("arabic-glm-ocr"),
    ):
        """Check that both OCR models are available via Ollama."""
        router = OCRRouter(ollama_url=ollama_url, en_model=en_model, ar_model=ar_model)
        status = router.health_check()
        for model, ok in status.items():
            icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
            console.print(f"{icon} {model}: {'available' if ok else 'NOT FOUND'}")
        if not all(status.values()):
            console.print("[yellow]Tip:[/yellow] Run [bold]ollama pull glm-ocr[/bold] to install missing models")
            raise typer.Exit(1)

    @app.command()
    def warmup(
        ollama_url: str = typer.Option("http://localhost:11434", help="Ollama server URL"),
        en_model: str = typer.Option("glm-ocr"),
        ar_model: str = typer.Option("arabic-glm-ocr"),
    ):
        """Pre-load OCR models into VRAM (eliminates first-request cold start).

        Run this once after `ollama serve` is up so your first real
        document doesn't time out on model load. Cold start can take
        60-120 s on consumer hardware; subsequent requests are fast.
        """
        console.print(
            f"[bold]Warming up:[/bold] [cyan]{en_model}[/cyan]"
            + (f" + [cyan]{ar_model}[/cyan]" if en_model != ar_model else " (single-model mode)")
        )
        router = OCRRouter(ollama_url=ollama_url, en_model=en_model, ar_model=ar_model)
        results = router.warmup()
        for model, ok in results.items():
            icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
            console.print(f"  {icon} {model}: {'loaded' if ok else 'FAILED'}")
        if not all(results.values()):
            console.print(
                "[yellow]One or more warmups failed.[/yellow] Check that:\n"
                "  - Ollama is running: [bold]curl http://localhost:11434/api/tags[/bold]\n"
                "  - The model is pulled: [bold]ollama pull glm-ocr[/bold]\n"
                "  - For very slow machines, set [bold]PHARMOCR_OCR_TIMEOUT=900[/bold]"
            )
            raise typer.Exit(1)

    @app.command()
    def batch(
        input_dir: Path = typer.Option(..., "--input-dir", help="Folder of PDF/image files"),
        output: Path = typer.Option(Path("./output"), "--output", "-o"),
        format: str = typer.Option("json", "--format", "-f"),
        ollama_url: str = typer.Option("http://localhost:11434"),
        confidence: float = typer.Option(0.85),
        en_model: str = typer.Option("glm-ocr"),
        ar_model: str = typer.Option("arabic-glm-ocr"),
    ):
        """Batch-process a folder of pharmaceutical leaflets."""
        supported = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}
        files = [f for f in input_dir.iterdir() if f.suffix.lower() in supported]
        if not files:
            console.print(f"[red]No supported files found in {input_dir}[/red]")
            raise typer.Exit(1)

        console.print(f"Found [bold]{len(files)}[/bold] files to process")
        ok = 0
        for f in files:
            try:
                _run_pipeline(
                    f, output, format, ollama_url,
                    confidence=confidence, en_model=en_model, ar_model=ar_model,
                )
                console.print(f"  [green]✓[/green] {f.name}")
                ok += 1
            except Exception as exc:
                console.print(f"  [red]✗[/red] {f.name}: {exc}")

        console.print(f"\n[bold green]Done:[/bold green] {ok}/{len(files)} files processed")


def main():
    if _RICH:
        app()
    else:
        print("Install typer and rich: pip install typer rich")
        sys.exit(1)


if __name__ == "__main__":
    main()
