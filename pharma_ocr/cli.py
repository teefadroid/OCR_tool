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
) -> dict:
    """Core pipeline runner. Returns result dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1
    preprocessor = DocumentPreprocessor(dpi=dpi)
    pages = preprocessor.process(input_path)

    # Stage 2
    layout = LayoutAnalyzer(mode="ollama", ollama_url=ollama_url)
    all_regions = []
    for page in pages:
        all_regions.extend(layout.analyze(page))

    # Stage 3
    router = OCRRouter(ollama_url=ollama_url)
    all_regions = router.process_regions(all_regions)

    # Stage 4
    postprocessor = PostProcessor(confidence_threshold=confidence)
    result = postprocessor.process(all_regions, source_file=str(input_path))

    # Stage 5
    json_exp = JSONExporter(output_dir)
    json_path = json_exp.export(result)

    if fmt in ("markdown", "both"):
        md_exp = MarkdownExporter(output_dir)
        md_exp.export(result)

    return json_exp.to_dict(result)


if _RICH:
    @app.command()
    def process(
        input: Path = typer.Option(..., "--input", "-i", help="PDF or image file"),
        output: Path = typer.Option(Path("./output"), "--output", "-o", help="Output directory"),
        format: str = typer.Option("json", "--format", "-f", help="json | markdown | both"),
        ollama_url: str = typer.Option("http://localhost:11434", help="Ollama server URL"),
        dpi: int = typer.Option(300, help="DPI for PDF rendering"),
        confidence: float = typer.Option(0.85, help="Confidence threshold for QC flags"),
    ):
        """Process a single pharmaceutical leaflet PDF or image."""
        if not input.exists():
            console.print(f"[red]Error:[/red] File not found: {input}")
            raise typer.Exit(1)

        console.print(f"[bold green]PharmOCR[/bold green] processing: [cyan]{input}[/cyan]")

        result = _run_pipeline(input, output, format, ollama_url, dpi, confidence)

        # Summary table
        t = Table(title="Extraction Summary")
        t.add_column("Field", style="bold")
        t.add_column("Value")
        t.add_row("Source", str(input))
        t.add_row("Pages", str(result.get("pages_processed", 0)))
        t.add_row("INN Matches", str(len(result.get("inn_matches", []))))
        t.add_row("Strengths Found", str(len(result.get("text_content", {}).get("combined", ""))))
        t.add_row("Regions for Review", str(len(result.get("ocr_metadata", {}).get("regions_flagged_for_review", []))))
        console.print(t)
        console.print(f"[green]✓ Output written to:[/green] {output}")

    @app.command()
    def health(
        ollama_url: str = typer.Option("http://localhost:11434", help="Ollama server URL"),
    ):
        """Check that both OCR models are available via Ollama."""
        router = OCRRouter(ollama_url=ollama_url)
        status = router.health_check()
        for model, ok in status.items():
            icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
            console.print(f"{icon} {model}: {'available' if ok else 'NOT FOUND'}")
        if not all(status.values()):
            console.print("[yellow]Tip:[/yellow] Run [bold]ollama pull glm-ocr[/bold] to install missing models")
            raise typer.Exit(1)

    @app.command()
    def batch(
        input_dir: Path = typer.Option(..., "--input-dir", help="Folder of PDF/image files"),
        output: Path = typer.Option(Path("./output"), "--output", "-o"),
        format: str = typer.Option("json", "--format", "-f"),
        ollama_url: str = typer.Option("http://localhost:11434"),
        confidence: float = typer.Option(0.85),
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
                _run_pipeline(f, output, format, ollama_url, confidence=confidence)
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
