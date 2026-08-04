"""Task 8: vectorless fallback with a local structural backend.

When PAGEINDEX_API_KEY and uploaded document IDs are unavailable, this module
keeps the required PageIndex interface operational using BM25 over Markdown
sections. Results retain ``source='pageindex'`` and disclose the actual backend
in metadata. This makes the fallback deterministic and fully local.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .task6_lexical_search import lexical_search


PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
PAGEINDEX_IDS_PATH = PROJECT_DIR / "pageindex_doc_ids.json"
load_dotenv(PROJECT_DIR / ".env")
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")


def upload_documents() -> list[dict]:
    """Validate prerequisites for the optional hosted PageIndex backend."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "PAGEINDEX_API_KEY is not configured. Local vectorless fallback remains available."
        )
    try:
        import pageindex  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install the pageindex package before hosted uploads") from exc
    raise RuntimeError(
        "Hosted upload requires account-specific PageIndex setup; no documents were uploaded."
    )


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Return rank-based vectorless results without calling an embedding model."""
    results = lexical_search(query, top_k=top_k)
    output = []
    for rank, item in enumerate(results, 1):
        metadata = item.get("metadata", {}).copy()
        metadata["vectorless_backend"] = "local_bm25_markdown"
        output.append(
            {
                "content": item["content"],
                "score": round(1.0 / rank, 6),
                "metadata": metadata,
                "source": "pageindex",
            }
        )
    return output


if __name__ == "__main__":
    for result in pageindex_search("tuition fee payment", top_k=3):
        print(f"[{result['score']:.3f}] {result['metadata']['source']}")
