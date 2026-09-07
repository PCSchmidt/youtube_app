"""FAISS vector store: IndexFlatIP over L2-normalized vectors.

Inner product on normalized vectors equals cosine similarity, which is the
documented retrieval metric for this project. Exact (flat) search is used
because portfolio-scale corpora are small; swapping in an ANN index later
changes nothing outside this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yt_rag.chunk import Chunk
from yt_rag.errors import IndexStateError


class FaissVectorStore:
    """Thin wrapper around faiss.IndexFlatIP carrying chunk metadata."""

    def __init__(self, dim: int) -> None:
        import faiss  # lazy: native library import

        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        vectors = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) vectors, got {vectors.shape}")
        if vectors.shape[0] != len(chunks):
            raise ValueError("one chunk per vector required")
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[float, Chunk]]:
        """Return the top-k (score, chunk) pairs, highest score first."""
        if len(self._chunks) == 0:
            raise IndexStateError("vector store is empty; ingest a transcript first")
        query = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        if query.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {query.shape[1]}")
        k = max(1, min(k, len(self._chunks)))
        scores, positions = self._index.search(query, k)
        return [
            (float(score), self._chunks[int(pos)])
            for score, pos in zip(scores[0], positions[0])
            if 0 <= pos < len(self._chunks)
        ]

    def save(self, directory: str | Path) -> Path:
        """Persist index + chunk metadata to a directory (artifacts path)."""
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        meta = [chunk.__dict__ for chunk in self._chunks]
        (directory / "chunks.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> FaissVectorStore:
        import faiss

        directory = Path(directory)
        index_path = directory / "index.faiss"
        if not index_path.is_file():
            raise IndexStateError(f"no saved index at {index_path}")
        store = cls.__new__(cls)
        store.dim = dim = int(faiss.read_index(str(index_path)).d)
        store._index = faiss.read_index(str(index_path))
        meta = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        store._chunks = [Chunk(**m) for m in meta]
        assert store._index.ntotal == len(store._chunks), "index/metadata mismatch"
        del dim
        return store
