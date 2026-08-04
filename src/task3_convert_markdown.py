"""Task 3 — Convert toàn bộ dữ liệu landing sang Markdown."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_ROOT / "data" / "landing"
OUTPUT_DIR = PROJECT_ROOT / "data" / "standardized"
LEGAL_EXTENSIONS = {".pdf", ".doc", ".docx"}
NEWS_EXTENSIONS = {".json", ".html", ".htm", ".txt", ".md"}


def setup_directories() -> None:
    (OUTPUT_DIR / "legal").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "news").mkdir(parents=True, exist_ok=True)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    text = str(value).replace("\x00", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def safe_title(value: Any, fallback: str) -> str:
    return re.sub(r"\s+", " ", clean_text(value) or fallback).strip()


def get_markitdown():
    try:
        from markitdown import MarkItDown
        return MarkItDown()
    except ImportError:
        return None


def legal_source_url(filename: str) -> str:
    try:
        from src.task1_collect_legal_docs import DOCUMENTS
    except (ImportError, ModuleNotFoundError):
        try:
            from task1_collect_legal_docs import DOCUMENTS
        except (ImportError, ModuleNotFoundError):
            return "N/A"
    value = DOCUMENTS.get(filename, "N/A")
    if isinstance(value, (tuple, list)):
        return str(value[0]) if value else "N/A"
    return str(value)


def convert_with_pdftotext(filepath: Path) -> str:
    executable = shutil.which("pdftotext")
    if not executable:
        raise RuntimeError('Hãy cài: python -m pip install "markitdown[pdf]"')
    process = subprocess.run(
        [executable, "-layout", "-enc", "UTF-8", str(filepath), "-"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return clean_text(process.stdout)


def extract_document_text(filepath: Path, converter) -> str:
    if converter is not None:
        result = converter.convert(str(filepath))
        return clean_text(getattr(result, "text_content", ""))
    if filepath.suffix.lower() == ".pdf":
        return convert_with_pdftotext(filepath)
    raise RuntimeError(f"MarkItDown là bắt buộc để convert {filepath.suffix}")


def write_markdown(output_path: Path, content: str) -> Path:
    content = clean_text(content)
    if not content:
        raise ValueError("Nội dung sau khi convert bị rỗng")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return output_path


def convert_legal_file(filepath: Path, converter) -> Path:
    text = extract_document_text(filepath, converter)
    title = safe_title(filepath.stem.replace("-", " ").replace("_", " "), filepath.stem)
    markdown = (
        f"# {title}\n\n"
        "## Metadata\n\n"
        f"- **Source:** {legal_source_url(filepath.name)}\n"
        f"- **Source file:** {filepath.name}\n"
        "- **Document type:** University policy/regulation\n\n"
        "---\n\n## Content\n\n"
        f"{text}\n"
    )
    return write_markdown(OUTPUT_DIR / "legal" / f"{filepath.stem}.md", markdown)


def get_first(data: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def json_to_markdown(filepath: Path) -> str:
    data = json.loads(filepath.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        return (
            f"# {filepath.stem}\n\n## Metadata\n\n"
            f"- **Source file:** {filepath.name}\n"
            "- **Document type:** University news/service information\n\n"
            "---\n\n## Content\n\n"
            f"```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```\n"
        )

    title = safe_title(get_first(data, ("title", "name", "heading"), filepath.stem), filepath.stem)
    url = clean_text(get_first(data, ("url", "source_url", "source"), "N/A"))
    crawled = clean_text(get_first(data, ("date_crawled", "crawled_at", "crawl_date"), "N/A"))
    published = clean_text(get_first(data, ("published_date", "date_published", "date"), ""))
    language = clean_text(get_first(data, ("language", "lang"), ""))
    content = clean_text(get_first(data, (
        "content_markdown", "markdown", "content", "text_content", "body", "article"
    ), ""))

    if not content:
        ignored = {"title", "name", "heading", "url", "source_url", "source", "date_crawled", "crawled_at", "crawl_date", "published_date", "date_published", "date", "language", "lang"}
        content = clean_text({k: v for k, v in data.items() if k not in ignored})

    metadata = [
        f"- **Source:** {url or 'N/A'}",
        f"- **Crawled:** {crawled or 'N/A'}",
        f"- **Source file:** {filepath.name}",
        "- **Document type:** University news/service information",
    ]
    if published:
        metadata.append(f"- **Published:** {published}")
    if language:
        metadata.append(f"- **Language:** {language}")

    return f"# {title}\n\n## Metadata\n\n" + "\n".join(metadata) + f"\n\n---\n\n## Content\n\n{content}\n"


def convert_news_file(filepath: Path, converter) -> Path:
    suffix = filepath.suffix.lower()
    if suffix == ".json":
        markdown = json_to_markdown(filepath)
    elif suffix in {".md", ".txt"}:
        raw = filepath.read_text(encoding="utf-8-sig")
        markdown = (
            f"# {safe_title(filepath.stem, filepath.stem)}\n\n"
            "## Metadata\n\n"
            f"- **Source file:** {filepath.name}\n"
            "- **Document type:** University news/service information\n\n"
            "---\n\n## Content\n\n"
            f"{clean_text(raw)}\n"
        )
    else:
        if converter is None:
            raise RuntimeError(f"Cần MarkItDown để convert {suffix}")
        result = converter.convert(str(filepath))
        text = clean_text(getattr(result, "text_content", ""))
        markdown = (
            f"# {safe_title(filepath.stem, filepath.stem)}\n\n"
            "## Metadata\n\n"
            f"- **Source file:** {filepath.name}\n"
            "- **Document type:** University news/service information\n\n"
            "---\n\n## Content\n\n"
            f"{text}\n"
        )
    return write_markdown(OUTPUT_DIR / "news" / f"{filepath.stem}.md", markdown)


def convert_legal_docs(converter=None) -> list[Path]:
    converter = converter if converter is not None else get_markitdown()
    legal_dir = LANDING_DIR / "legal"
    if not legal_dir.exists():
        print(f"⚠ Không tìm thấy: {legal_dir}")
        return []
    outputs: list[Path] = []
    for filepath in sorted(legal_dir.rglob("*")):
        if not filepath.is_file() or filepath.suffix.lower() not in LEGAL_EXTENSIONS:
            continue
        try:
            output = convert_legal_file(filepath, converter)
            outputs.append(output)
            print(f"✓ Legal: {filepath.name} -> {output.name}")
        except Exception as exc:
            print(f"✗ Legal lỗi: {filepath.name}: {exc}")
    return outputs


def convert_news_articles(converter=None) -> list[Path]:
    converter = converter if converter is not None else get_markitdown()
    news_dir = LANDING_DIR / "news"
    if not news_dir.exists():
        print(f"⚠ Không tìm thấy: {news_dir}")
        return []
    outputs: list[Path] = []
    for filepath in sorted(news_dir.rglob("*")):
        if not filepath.is_file() or filepath.suffix.lower() not in NEWS_EXTENSIONS:
            continue
        try:
            output = convert_news_file(filepath, converter)
            outputs.append(output)
            print(f"✓ News: {filepath.name} -> {output.name}")
        except Exception as exc:
            print(f"✗ News lỗi: {filepath.name}: {exc}")
    return outputs


def validate_outputs(outputs: list[Path]) -> None:
    legal = [p for p in outputs if p.parent.name == "legal"]
    news = [p for p in outputs if p.parent.name == "news"]
    short_files = [p for p in outputs if len(p.read_text(encoding="utf-8")) <= 200]
    print("\n" + "=" * 60)
    print("TASK 3 SUMMARY")
    print("=" * 60)
    print(f"Legal Markdown : {len(legal)}")
    print(f"News Markdown  : {len(news)}")
    print(f"Total          : {len(outputs)}")
    print(f"Output folder  : {OUTPUT_DIR}")
    if len(legal) < 3:
        print("⚠ Chưa đủ 3 file legal Markdown.")
    if len(news) < 5:
        print("⚠ Chưa đủ 5 file news Markdown.")
    if short_files:
        print("⚠ File có nội dung <= 200 ký tự:")
        for path in short_files:
            print(f"  - {path}")
    elif outputs:
        print("✓ Dữ liệu Markdown đã sẵn sàng cho Task 4.")


def convert_all() -> list[Path]:
    setup_directories()
    converter = get_markitdown()
    if converter is None:
        print('⚠ Chưa cài MarkItDown. Chạy: python -m pip install "markitdown[pdf,docx]"')
    outputs = convert_legal_docs(converter)
    outputs.extend(convert_news_articles(converter))
    validate_outputs(outputs)
    return outputs


def main() -> None:
    outputs = convert_all()
    if not outputs:
        raise SystemExit("Không tạo được file Markdown nào. Hãy kiểm tra data/landing/.")


if __name__ == "__main__":
    main()
