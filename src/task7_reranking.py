"""Task 7: local MMR and Reciprocal Rank Fusion reranking."""

from __future__ import annotations

import math


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Optional local proxy reranker; RRF is the selected production method."""
    query_terms = set(query.casefold().split())
    output = []
    for item in candidates:
        content_terms = set(item["content"].casefold().split())
        overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
        rescored = item.copy()
        rescored["score"] = round(0.7 * overlap + 0.3 * float(item.get("score", 0.0)), 6)
        output.append(rescored)
    return sorted(output, key=lambda item: item["score"], reverse=True)[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must be between 0 and 1")
    if any("embedding" not in item for item in candidates):
        raise ValueError("MMR candidates must contain embeddings")
    selected: list[int] = []
    selected_scores: list[float] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < top_k:
        best_index = remaining[0]
        best_score = float("-inf")
        for index in remaining:
            relevance = _cosine(query_embedding, candidates[index]["embedding"])
            redundancy = max(
                (_cosine(candidates[index]["embedding"], candidates[chosen]["embedding"])
                 for chosen in selected),
                default=0.0,
            )
            score = lambda_param * relevance - (1.0 - lambda_param) * redundancy
            if score > best_score:
                best_index, best_score = index, score
        result = candidates[best_index].copy()
        result["score"] = round(best_score, 6)
        selected.append(best_index)
        selected_scores.append(best_score)
        remaining.remove(best_index)
    return [
        {**candidates[index], "score": round(score, 6)}
        for index, score in zip(selected, selected_scores)
    ]


def _identity(item: dict) -> str:
    metadata = item.get("metadata", {})
    return f"{metadata.get('source_path', metadata.get('source', ''))}:{metadata.get('chunk_index', '')}:{item['content']}"


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    if top_k <= 0:
        return []
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        seen: set[str] = set()
        for rank, item in enumerate(ranked_list, 1):
            key = _identity(item)
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items.setdefault(key, item)
    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    output = []
    for key in ranked_keys[:top_k]:
        item = items[key].copy()
        item["score"] = round(scores[key], 8)
        output.append(item)
    return output


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k=top_k)
    raise ValueError("Use rerank_mmr directly for MMR, because it requires embeddings")


if __name__ == "__main__":
    demo = [{"content": "Tuition fee policy", "score": 0.8, "metadata": {}},
            {"content": "Library rooms", "score": 0.5, "metadata": {}}]
    print(rerank("tuition fee", demo, top_k=2))
