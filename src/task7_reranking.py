"""
Task 7 (mở rộng) — Jina Cross-Encoder Reranker với RRF fallback.

- Có JINA_API_KEY: gọi Jina Reranker API.
- Không có key hoặc API lỗi: fallback về Reciprocal Rank Fusion (RRF).

Chạy:
    python -m src.task7_reranking

Test:
    python -m pytest tests/test_individual.py::TestTask7 -v
"""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()

JINA_API_URL = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = os.getenv(
    "JINA_RERANK_MODEL",
    "jina-reranker-v2-base-multilingual",
)
JINA_TIMEOUT_SECONDS = float(os.getenv("JINA_TIMEOUT_SECONDS", "30"))
RRF_K = 60


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")


def _normalise_candidates(candidates: list[dict]) -> list[dict]:
    """Validate and copy candidates without mutating caller data."""
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list")

    normalised: list[dict] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise TypeError(f"candidate at index {index} must be a dict")

        content = str(candidate.get("content", "")).strip()
        if not content:
            continue

        item = candidate.copy()
        item["content"] = content
        item.setdefault("metadata", {})
        item.setdefault("score", 0.0)
        normalised.append(item)

    return normalised


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = RRF_K,
) -> list[dict]:
    """
    Gộp một hoặc nhiều danh sách xếp hạng bằng Reciprocal Rank Fusion.

    RRF score(d) = sum(1 / (k + rank))

    Khóa nhận diện ưu tiên metadata chunk_id/id; nếu không có thì dùng content.
    """
    _validate_top_k(top_k)
    if k < 0:
        raise ValueError("k must be non-negative")
    if not ranked_lists:
        return []

    scores: dict[str, float] = {}
    item_map: dict[str, dict] = {}
    first_seen: dict[str, int] = {}
    sequence = 0

    for ranked_list in ranked_lists:
        for rank, raw_item in enumerate(ranked_list, start=1):
            if not isinstance(raw_item, dict):
                continue

            content = str(raw_item.get("content", "")).strip()
            if not content:
                continue

            metadata = raw_item.get("metadata") or {}
            identity = (
                metadata.get("chunk_id")
                or metadata.get("id")
                or raw_item.get("id")
                or content
            )
            key = str(identity)

            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in item_map:
                item_map[key] = raw_item.copy()
                first_seen[key] = sequence
                sequence += 1

    ordered_keys = sorted(
        scores,
        key=lambda key: (-scores[key], first_seen[key]),
    )

    output: list[dict] = []
    for key in ordered_keys[:top_k]:
        item = item_map[key].copy()
        item["original_score"] = item.get("score", 0.0)
        item["score"] = round(scores[key], 8)
        item["rerank_method"] = "rrf"
        item.setdefault("metadata", {})
        output.append(item)

    return output


def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    *,
    api_key: str | None = None,
) -> list[dict]:
    """Rerank candidates bằng Jina multilingual cross-encoder API."""
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    _validate_top_k(top_k)

    candidates = _normalise_candidates(candidates)
    if not candidates:
        return []

    key = api_key or os.getenv("JINA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("JINA_API_KEY is not configured")

    response = requests.post(
        JINA_API_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "model": JINA_RERANK_MODEL,
            "query": query,
            "documents": [item["content"] for item in candidates],
            "top_n": min(top_k, len(candidates)),
            "return_documents": False,
        },
        timeout=JINA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    payload: dict[str, Any] = response.json()
    api_results = payload.get("results")
    if not isinstance(api_results, list):
        raise RuntimeError("Jina response does not contain a valid 'results' list")

    output: list[dict] = []
    for result in api_results:
        index = result.get("index")
        relevance_score = result.get("relevance_score")

        if not isinstance(index, int) or not 0 <= index < len(candidates):
            continue
        if relevance_score is None:
            continue

        item = candidates[index].copy()
        item["original_score"] = item.get("score", 0.0)
        item["score"] = float(relevance_score)
        item["rerank_method"] = "jina_cross_encoder"
        item["rerank_model"] = JINA_RERANK_MODEL
        item.setdefault("metadata", {})
        output.append(item)

    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "auto",
) -> list[dict]:
    """
    Unified interface dùng cho Task 7 và Task 9.

    method:
        - "auto": có JINA_API_KEY thì dùng Jina; nếu thiếu key/API lỗi thì RRF.
        - "cross_encoder" hoặc "jina": bắt buộc gọi Jina; lỗi sẽ được ném ra.
        - "rrf": dùng RRF local, không cần API key.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    _validate_top_k(top_k)

    candidates = _normalise_candidates(candidates)
    if not candidates:
        return []

    method = method.strip().lower()

    if method in {"cross_encoder", "jina"}:
        return rerank_cross_encoder(query, candidates, top_k)

    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)

    if method != "auto":
        raise ValueError(
            "Unknown rerank method. Use: auto, jina, cross_encoder, or rrf"
        )

    if os.getenv("JINA_API_KEY", "").strip():
        try:
            results = rerank_cross_encoder(query, candidates, top_k)
            if results:
                return results
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            print(f"Warning: Jina reranker failed; fallback to RRF: {exc}")

    return rerank_rrf([candidates], top_k=top_k)


if __name__ == "__main__":
    dummy_candidates = [
        {
            "content": "Tuition fee payment schedule and accepted payment methods",
            "score": 0.80,
            "metadata": {"source": "tuition-fees.md"},
        },
        {
            "content": "Scholarship eligibility requirements for RMIT students",
            "score": 0.60,
            "metadata": {"source": "scholarships.md"},
        },
        {
            "content": "Library study room booking guide",
            "score": 0.50,
            "metadata": {"source": "library.md"},
        },
    ]

    results = rerank(
        "How can I pay my tuition fee?",
        dummy_candidates,
        top_k=2,
    )

    for result in results:
        print(
            f"[{result['score']:.4f}] "
            f"[{result['rerank_method']}] "
            f"{result['content']}"
        )