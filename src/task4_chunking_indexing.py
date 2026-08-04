"""Task 4: chunk Markdown, embed with OpenAI, and persist in local ChromaDB."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

# Character-based recursive-style splitting is predictable for the mixed
# English/Vietnamese Markdown corpus. 1,000 characters usually preserves one
# compact policy section; 150 characters (15%) keeps context across boundaries
# without creating excessive duplicate vectors.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNKING_METHOD = "recursive_character"

# text-embedding-3-small returns 1,536 dimensions by default. We pass the
# dimension explicitly so indexing and query embedding cannot drift apart.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBEDDING_BATCH_SIZE = 64

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"


def load_documents() -> list[dict]:
    """Load every non-empty Markdown document with retrieval metadata."""
    documents: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        source_match = re.search(r"^\*\*Source:\*\*\s*(.+)$", content, re.MULTILINE)
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "source_path": relative_path,
                    "source_url": source_match.group(1).strip() if source_match else "N/A",
                    "type": "legal" if relative_path.startswith("legal/") else "news",
                },
            }
        )
    return documents


def _split_text(text: str) -> list[str]:
    """Split near Markdown/paragraph boundaries while enforcing a hard limit."""
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    separators = ("\n## ", "\n### ", "\n\n", "\n", ". ", "; ", " ")
    while start < len(text):
        hard_end = min(start + CHUNK_SIZE, len(text))
        end = hard_end
        if hard_end < len(text):
            search_from = start + max(CHUNK_OVERLAP + 1, int(CHUNK_SIZE * 0.55))
            for separator in separators:
                boundary = text.rfind(separator, search_from, hard_end)
                if boundary >= search_from:
                    end = boundary + (1 if separator.startswith("\n") else len(separator))
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(0, end - CHUNK_OVERLAP)
        start = next_start if next_start > start else end
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Create deterministic chunks and inherit document metadata."""
    chunks: list[dict] = []
    for document in documents:
        for chunk_index, content in enumerate(_split_text(document["content"])):
            chunks.append(
                {
                    "content": content,
                    "metadata": {**document["metadata"], "chunk_index": chunk_index},
                }
            )
    return chunks


def _batched(items: list, size: int) -> Iterable[list]:
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def get_openai_client():
    """Create an SDK client without exposing or persisting the API key."""
    load_dotenv(PROJECT_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or the environment.")
    from openai import OpenAI

    return OpenAI()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches using the configured OpenAI embedding model."""
    if not texts:
        return []
    client = get_openai_client()
    embeddings: list[list[float]] = []
    for batch in _batched(texts, EMBEDDING_BATCH_SIZE):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIM,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        embeddings.extend(item.embedding for item in ordered)
    if len(embeddings) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: {len(embeddings)} != {len(texts)}")
    return embeddings


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Return chunks augmented with 1,536-dimensional embedding vectors."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    embedded: list[dict] = []
    for chunk, vector in zip(chunks, vectors):
        if len(vector) != EMBEDDING_DIM:
            raise RuntimeError(f"Unexpected embedding dimension: {len(vector)}")
        embedded.append({**chunk, "embedding": vector})
    return embedded


def get_chroma_client():
    import chromadb
    from chromadb.config import Settings

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection():
    """Open the existing local collection used by Task 5."""
    return get_chroma_client().get_collection(name=COLLECTION_NAME)


def _chunk_id(chunk: dict) -> str:
    metadata = chunk["metadata"]
    identity = f"{metadata['source_path']}:{metadata['chunk_index']}:{chunk['content']}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def index_to_vectorstore(chunks: list[dict], *, reset: bool = True):
    """Persist embedded chunks locally; reset avoids stale corpus fragments."""
    if not chunks:
        raise ValueError("No chunks to index.")
    if any("embedding" not in chunk for chunk in chunks):
        raise ValueError("Every chunk must contain an embedding before indexing.")

    client = get_chroma_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception as exc:
            # Chroma versions use different not-found exception classes.
            if "does not exist" not in str(exc).lower() and "not found" not in str(exc).lower():
                raise

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIM,
            "chunking_method": CHUNKING_METHOD,
        },
    )
    collection.upsert(
        ids=[_chunk_id(chunk) for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return collection


def run_pipeline():
    """Run load -> chunk -> OpenAI embeddings -> local Chroma indexing."""
    print("=" * 60)
    print(f"Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"Embedding: {EMBEDDING_MODEL} ({EMBEDDING_DIM} dimensions)")
    print(f"Chroma path: {CHROMA_DIR}")
    print("=" * 60)

    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks")
    collection = index_to_vectorstore(embedded_chunks, reset=True)
    print(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}'")
    return collection


if __name__ == "__main__":
    run_pipeline()
