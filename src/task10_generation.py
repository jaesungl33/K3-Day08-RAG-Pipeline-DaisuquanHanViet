"""Task 10: grounded answer generation with inline source citations."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


PROJECT_DIR = Path(__file__).parent.parent
load_dotenv(PROJECT_DIR / ".env")

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.2
# Cost-aware default for repeated RAG evaluation. Override without code changes.
LLM_MODEL = os.getenv("OPENAI_GENERATION_MODEL", "gpt-4o-mini")

INSUFFICIENT_EVIDENCE = "I cannot verify this information from the available sources."
SYSTEM_PROMPT = f"""You are an assistant for RMIT Vietnam policies and student services.
Use only the documents supplied in CONTEXT. Do not use external knowledge, infer missing facts,
or guess.

Every sentence containing a factual claim must include at least one inline Markdown citation in
the exact form [Source, Year](Source URL). Use the Citation label and Source URL from the same
document in CONTEXT. Copy supplied URLs exactly; never invent, alter, or infer a URL.
Do not use a document whose Source URL is N/A as evidence for a factual claim.

End every substantive answer with a "References" section listing only sources actually cited,
using one bullet per source in the form:
- [Source, Year](Source URL)

If CONTEXT does not contain sufficient evidence with a valid URL, respond with exactly:
{INSUFFICIENT_EVIDENCE}

Answer in English. Be direct, clear, concise, and fully grounded in CONTEXT."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    if len(chunks) <= 2:
        return list(chunks)
    return list(chunks[::2]) + list(chunks[1::2])[::-1]


def _citation_label(chunk: dict, index: int) -> str:
    source = chunk.get("metadata", {}).get("source", f"Source {index}")
    clean_source = Path(source).stem.replace("-rmit", "").replace("-", " ").title()
    year_match = re.search(r"(?:19|20)\d{2}", source)
    year = year_match.group(0) if year_match else "n.d."
    return f"{clean_source}, {year}"


def format_context(chunks: list[dict]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        citation = _citation_label(chunk, index)
        parts.append(
            f"[Document {index}]\n"
            f"Citation label: [{citation}]\n"
            f"Source file: {metadata.get('source', 'Unknown')}\n"
            f"Source URL: {metadata.get('source_url', 'N/A')}\n"
            f"Content:\n{chunk['content'].strip()}"
        )
    return "\n\n---\n\n".join(parts)


def _generate_answer(query: str, context: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env or the environment")
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}",
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return (response.choices[0].message.content or INSUFFICIENT_EVIDENCE).strip()


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    *,
    retrieval_mode: str = "hybrid",
    use_reranking: bool = True,
) -> dict:
    chunks = retrieve(
        query,
        top_k=top_k,
        use_reranking=use_reranking,
        mode=retrieval_mode,
    )
    if not chunks:
        return {"answer": INSUFFICIENT_EVIDENCE, "sources": [], "retrieval_source": "none"}
    reordered = reorder_for_llm(chunks)
    answer = _generate_answer(query, format_context(reordered))
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    result = generate_with_citation("How can I book a library study room?")
    print(result["answer"])
