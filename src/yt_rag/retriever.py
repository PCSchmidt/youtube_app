"""Top-k retrieval over the FAISS store."""

from __future__ import annotations

from dataclasses import dataclass

from yt_rag.chunk import Chunk
from yt_rag.embeddings import EmbeddingProvider
from yt_rag.vectorstore import FaissVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk retrieved for a query, with its cosine similarity score."""

    score: float
    chunk: Chunk


class Retriever:
    """Embeds the query and returns the top-k chunks by cosine similarity."""

    def __init__(self, store: FaissVectorStore, embedder: EmbeddingProvider, k: int = 4):
        if store.dim != embedder.dim:
            raise ValueError(f"embedder dim {embedder.dim} != index dim {store.dim}")
        self.store = store
        self.embedder = embedder
        self.k = k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed([query])[0]
        return [
            RetrievedChunk(score=score, chunk=chunk)
            for score, chunk in self.store.search(query_vector, self.k)
        ]
