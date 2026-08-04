"""Convert all Task 1 and Task 2 landing files to Markdown."""

import json
import shutil
import subprocess
from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> list[Path]:
    """Convert documents with MarkItDown, or pdftotext when unavailable."""
    try:
        from markitdown import MarkItDown

        converter = MarkItDown()
    except ImportError:
        converter = None
        if not shutil.which("pdftotext"):
            raise RuntimeError(
                "Install markitdown[pdf], or make pdftotext available on PATH."
            )
        print("MarkItDown is unavailable; using local pdftotext fallback.")

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue
        print(f"Converting: {filepath.name}")
        if converter is not None:
            text = converter.convert(str(filepath)).text_content
        elif filepath.suffix.lower() == ".pdf":
            process = subprocess.run(
                ["pdftotext", "-layout", "-enc", "UTF-8", str(filepath), "-"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            text = process.stdout
        else:
            raise RuntimeError(f"MarkItDown is required to convert {filepath.suffix}: {filepath}")
        output_path = output_dir / f"{filepath.stem}.md"
        source_url = _legal_source_url(filepath.name)
        header = f"# {filepath.stem.replace('-', ' ').title()}\n\n**Source:** {source_url}\n\n---\n\n"
        output_path.write_text(header + text.strip() + "\n", encoding="utf-8")
        outputs.append(output_path)
        print(f"Saved: {output_path}")
    return outputs


def _legal_source_url(filename: str) -> str:
    try:
        from src.task1_collect_legal_docs import DOCUMENTS
    except ModuleNotFoundError:
        from task1_collect_legal_docs import DOCUMENTS

    return DOCUMENTS.get(filename, "N/A")


def convert_news_articles() -> list[Path]:
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for filepath in sorted(news_dir.glob("*.json")):
        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output_path = output_dir / f"{filepath.stem}.md"
        content = (
            f"# {data.get('title', 'Unknown')}\n\n"
            f"**Source:** {data.get('url', 'N/A')}\n\n"
            f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
            f"{data.get('content_markdown', '').strip()}\n"
        )
        output_path.write_text(content, encoding="utf-8")
        outputs.append(output_path)
        print(f"Saved: {output_path}")
    return outputs


def convert_all() -> list[Path]:
    outputs = convert_legal_docs()
    outputs.extend(convert_news_articles())
    print(f"Done: {len(outputs)} Markdown files in {OUTPUT_DIR}")
    return outputs


if __name__ == "__main__":
    convert_all()
