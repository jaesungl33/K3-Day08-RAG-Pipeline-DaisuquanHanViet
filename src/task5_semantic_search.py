"""Task 5: semantic search over the local ChromaDB collection."""

from __future__ import annotations

try:
    from .task4_chunking_indexing import embed_texts, get_collection
except ImportError:  # Support: python src/task5_semantic_search.py
    from task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return Chroma cosine matches sorted by similarity descending."""
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    query_vector = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    output: list[dict] = []
    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # The collection uses cosine distance, so similarity = 1 - distance.
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        output.append(
            {
                "content": document,
                "score": round(score, 6),
                "metadata": metadata,
            }
        )

    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    for result in semantic_search("What is the tuition fee payment policy?", top_k=5):
        print(f"[{result['score']:.4f}] {result['metadata']['source']}")
