"""Task 9: dense + BM25 hybrid retrieval with RRF and vectorless fallback."""

from __future__ import annotations

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.30
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    mode: str = "hybrid",
) -> list[dict]:
    """Retrieve with ``hybrid`` or ``dense_only`` configuration.

    Fallback is decided from the original dense cosine score, never the RRF
    score, because RRF expresses rank agreement rather than relevance.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if mode not in {"hybrid", "dense_only"}:
        raise ValueError("mode must be 'hybrid' or 'dense_only'")

    candidate_count = max(top_k * 2, 10)
    dense_results = semantic_search(query, top_k=candidate_count)
    best_dense_score = dense_results[0]["score"] if dense_results else 0.0

    if best_dense_score < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    if mode == "dense_only":
        final_results = [item.copy() for item in dense_results[:top_k]]
    else:
        sparse_results = lexical_search(query, top_k=candidate_count)
        if use_reranking:
            final_results = rerank_rrf(
                [dense_results, sparse_results], top_k=top_k
            )
        else:
            # Ablation without fusion reranking: concatenate and deduplicate,
            # preserving dense-first ordering.
            final_results = []
            seen = set()
            for item in dense_results + sparse_results:
                key = (item["metadata"].get("source_path"), item["metadata"].get("chunk_index"))
                if key not in seen:
                    seen.add(key)
                    final_results.append(item.copy())
                if len(final_results) >= top_k:
                    break

    for item in final_results:
        item["source"] = "hybrid"
        item["metadata"] = item.get("metadata", {}).copy()
        item["metadata"]["retrieval_mode"] = mode
        item["metadata"]["best_dense_score"] = best_dense_score
    return final_results[:top_k]


if __name__ == "__main__":
    for result in retrieve("How do I book a library study room?", top_k=3):
        print(f"[{result['score']:.4f}] {result['metadata']['source']}")
