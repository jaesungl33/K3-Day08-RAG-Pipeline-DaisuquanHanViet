"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học từ RMIT Vietnam.

Yêu cầu:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/.
    4. Mỗi bài lưu 1 file JSON với metadata: url, title, date_crawled, content.

Cài đặt:
    pip install crawl4ai
    crawl4ai-setup

Nếu lệnh trên không tải được Chromium, chạy thêm:
    playwright install chromium
"""

import asyncio
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "news"

# Các URL công khai, cùng thuộc hệ thống RMIT Vietnam.
# Nhóm dữ liệu gồm: tin thư viện, sự kiện, công cụ học tập và hỗ trợ sinh viên.
ARTICLE_URLS = [
    "https://www.rmit.edu.vn/libraryvn/about-us/news/2025/10-years-book-swap",
    "https://www.rmit.edu.vn/libraryvn/about-us/news/2025/rmit-vietnam-library-launches-adobe-express-champions",
    "https://www.rmit.edu.vn/libraryvn/about-us/library-events/2026/library-friends",
    "https://www.rmit.edu.vn/libraryvn/about-us/library-events/2026/world-questival",
    "https://www.rmit.edu.vn/students/student-news-and-events/student-events-2026/orientation-week-sem-2",
    "https://www.rmit.edu.vn/students/student-news-and-events/student-news/2026/discover-val-latest-updates-for-study",
    "https://www.rmit.edu.vn/students/student-news-and-events/student-news/2026/redefining-fairness-equitable-learning-accessibility",
]


def setup_directory() -> None:
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {DATA_DIR}")


def slugify(text: str, max_length: int = 80) -> str:
    """Chuyển tiêu đề thành tên file ASCII an toàn."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return (slug[:max_length].rstrip("-") or "untitled")


def extract_markdown(result: Any) -> str:
    """Đọc markdown tương thích với nhiều phiên bản Crawl4AI."""
    markdown = getattr(result, "markdown", "")

    if isinstance(markdown, str):
        return markdown.strip()

    # Crawl4AI mới trả về MarkdownGenerationResult.
    for attr in ("fit_markdown", "raw_markdown", "markdown_with_citations"):
        value = getattr(markdown, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return str(markdown).strip() if markdown else ""


async def crawl_article(url: str, crawler: Any | None = None) -> dict:
    """Crawl một bài viết và trả về metadata cùng nội dung Markdown."""
    from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=60_000,
        wait_until="domcontentloaded",
        remove_overlay_elements=True,
        excluded_tags=["nav", "footer", "script", "style", "noscript"],
        exclude_external_links=True,
        word_count_threshold=10,
    )

    owns_crawler = crawler is None
    if owns_crawler:
        crawler = AsyncWebCrawler()
        await crawler.start()

    try:
        result = await crawler.arun(url=url, config=run_config)

        if not getattr(result, "success", False):
            error = getattr(result, "error_message", "Unknown crawl error")
            raise RuntimeError(f"Crawl failed for {url}: {error}")

        metadata = getattr(result, "metadata", {}) or {}
        content = extract_markdown(result)
        if not content:
            raise RuntimeError(f"No content extracted from {url}")

        title = str(metadata.get("title") or "Unknown").strip()
        description = str(metadata.get("description") or "").strip()

        return {
            "url": url,
            "title": title,
            "description": description,
            "date_crawled": datetime.now(timezone.utc).isoformat(),
            "content_markdown": content,
        }
    finally:
        if owns_crawler:
            await crawler.close()


async def crawl_all() -> None:
    """Crawl toàn bộ URL; lỗi một bài không làm dừng cả chương trình."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig

    setup_directory()
    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium",
        verbose=False,
    )

    success_count = 0
    failed_urls: list[str] = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for i, url in enumerate(ARTICLE_URLS, start=1):
            print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")

            try:
                article = await crawl_article(url, crawler)
                filename = f"article_{i:02d}_{slugify(article['title'])}.json"
                filepath = DATA_DIR / filename
                filepath.write_text(
                    json.dumps(article, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                success_count += 1
                print(f"  ✓ Saved: {filepath.name}")
            except Exception as exc:
                failed_urls.append(url)
                print(f"  ✗ Error: {exc}")

    print(f"\nCompleted: {success_count}/{len(ARTICLE_URLS)} articles saved.")
    if failed_urls:
        print("Failed URLs:")
        for url in failed_urls:
            print(f"  - {url}")

    if success_count < 5:
        raise RuntimeError(
            "Fewer than 5 articles were saved. Check internet access, URL status, "
            "and whether Chromium was installed with `crawl4ai-setup`."
        )


if __name__ == "__main__":
    if len(ARTICLE_URLS) < 5:
        raise ValueError("ARTICLE_URLS must contain at least 5 public article URLs.")
    asyncio.run(crawl_all())
