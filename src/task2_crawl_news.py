"""Crawl public RMIT Vietnam student-service pages into JSON files."""

import html
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://www.rmit.edu.vn/study-at-rmit/international-students/international-student-support-services",
    "https://www.rmit.edu.vn/libraryvn/borrowing-and-resources/borrowing-and-returning",
    "https://www.rmit.edu.vn/libraryvn/student-support/book-a-study-room",
    "https://www.rmit.edu.vn/students/support/student-academic-success",
    "https://www.rmit.edu.vn/student-life/support-services/wellbeing",
]


class _MainContentParser(HTMLParser):
    """Small dependency-free extractor that preserves basic Markdown structure."""

    SKIP_TAGS = {"script", "style", "svg", "noscript", "nav", "footer"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.main_depth = 0
        self.saw_main = False
        self.heading: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if tag == "main":
            if not self.saw_main:
                # Discard header/navigation text collected before <main>.
                self.parts.clear()
            self.saw_main = True
            self.main_depth += 1
        if self.skip_depth or (self.saw_main and not self.main_depth):
            return
        if re.fullmatch(r"h[1-6]", tag):
            self.heading = tag
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in {"p", "div", "section", "article", "br", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if tag == "main" and self.main_depth:
            self.main_depth -= 1
        if tag == self.heading:
            self.parts.append("\n")
            self.heading = None

    def handle_data(self, data: str) -> None:
        if self.skip_depth or (self.saw_main and not self.main_depth):
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(text + " ")

    def markdown(self) -> str:
        content = "".join(self.parts)
        content = re.sub(r"[ \t]+\n", "\n", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return html.unescape(content).strip()


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def crawl_article(url: str) -> dict:
    response = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    # RMIT serves UTF-8 HTML but some responses omit the charset header, causing
    # requests to default to ISO-8859-1 and corrupt Vietnamese text.
    response.encoding = response.apparent_encoding or "utf-8"

    title_match = re.search(r"<title[^>]*>(.*?)</title>", response.text, re.I | re.S)
    title = html.unescape(re.sub(r"<[^>]+>", "", title_match.group(1))).strip() if title_match else "Unknown"
    title = re.sub(r"\s*[-|]\s*RMIT University\s*$", "", title, flags=re.I)

    parser = _MainContentParser()
    parser.feed(response.text)
    content = parser.markdown()
    # Adobe/RMIT pages do not consistently expose a semantic <main> element.
    # The real article heading is repeated after the global navigation, so keep
    # content from the final exact-title heading onward.
    heading_pattern = re.compile(
        rf"^#{{1,6}}\s+{re.escape(title)}\s*$", re.I | re.M
    )
    heading_matches = list(heading_pattern.finditer(content))
    if heading_matches:
        content = content[heading_matches[-1].start():].strip()
    if len(content) < 500:
        raise ValueError(f"Extracted content is unexpectedly short ({len(content)} chars): {url}")

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content,
    }


def crawl_all() -> list[Path]:
    setup_directory()
    outputs = []
    for index, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = crawl_article(url)
        filepath = DATA_DIR / f"article_{index:02d}.json"
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append(filepath)
        print(f"Saved: {filepath}")
    return outputs


if __name__ == "__main__":
    crawl_all()
