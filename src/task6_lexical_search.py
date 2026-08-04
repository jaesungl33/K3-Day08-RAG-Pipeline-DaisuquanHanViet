"""Task 6: dependency-free BM25 lexical retrieval over the Task 4 chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from .task4_chunking_indexing import chunk_documents, load_documents


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


class BM25Index:
    def __init__(self, corpus: list[dict], k1: float = BM25_K1, b: float = BM25_B):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(item["content"]) for item in corpus]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        total = len(corpus)
        self.idf = {
            term: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for frequencies, length in zip(self.term_frequencies, self.lengths):
            score = 0.0
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                norm = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / max(self.avg_length, 1.0)
                )
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / norm
            scores.append(score)
        return scores


def build_bm25_index(corpus: list[dict]) -> BM25Index:
    return BM25Index(corpus)


@lru_cache(maxsize=1)
def _get_corpus_and_index() -> tuple[list[dict], BM25Index]:
    corpus = chunk_documents(load_documents())
    return corpus, build_bm25_index(corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    corpus, index = _get_corpus_and_index()
    scores = index.get_scores(tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    results = []
    for index_value in ranked_indices:
        if scores[index_value] <= 0:
            break
        item = corpus[index_value]
        results.append(
            {
                "content": item["content"],
                "score": round(float(scores[index_value]), 6),
                "metadata": item["metadata"].copy(),
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("tuition fee payment methods", top_k=5):
        print(f"[{result['score']:.3f}] {result['metadata']['source']}")
