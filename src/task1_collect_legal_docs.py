"""Download public RMIT Vietnam policy/service documents for Task 1."""

from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

DOCUMENTS = {
    "student-fees-and-charges-guide-rmit-2026.pdf": (
        "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/documents/"
        "pdfs/study-at-rmit/tuition-fees/student-fees-and-charges-guide-06-2026.pdf"
    ),
    "international-student-guide-rmit-2026.pdf": (
        "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/documents/"
        "pdfs/study-at-rmit/international-students/international-student-guide-2026.pdf"
    ),
    "study-room-booking-instructions-rmit-2025.pdf": (
        "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/documents/"
        "pdfs/library/en/study-room-booking-instruction-2025.pdf"
    ),
}


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, filename: str) -> Path:
    """Download one public PDF and reject HTML/error responses."""
    response = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if not response.content.startswith(b"%PDF") and "application/pdf" not in content_type:
        raise ValueError(f"Expected PDF but received {content_type or 'unknown content'}: {url}")

    filepath = DATA_DIR / filename
    filepath.write_bytes(response.content)
    return filepath


def collect_all() -> list[Path]:
    setup_directory()
    downloaded = []
    for filename, url in DOCUMENTS.items():
        filepath = download_file(url, filename)
        downloaded.append(filepath)
        print(f"Saved: {filepath} ({filepath.stat().st_size:,} bytes)")
    return downloaded


if __name__ == "__main__":
    collect_all()
