"""Embedding providers.

Two implementations of one small interface:

- :class:`HashEmbedder`: deterministic, dependency-free, offline. Used by the
  default test suite so ``make test`` never downloads model weights.
- :class:`SentenceTransformerEmbedder`: the real, pinned model. Loaded lazily
  and only in the production path (see README for how to run it).

Both return L2-normalized float32 vectors, which makes inner product equal to
cosine similarity in the FAISS index.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

import numpy as np

from yt_rag.config import (
    EMBEDDING_MODEL_DIM,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
)


class EmbeddingProvider(Protocol):
    """Anything that can turn a list of texts into normalized vectors."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class HashEmbedder:
    """Deterministic offline embedder.

    A hashed bag-of-words: each whitespace token is hashed to ``dim`` buckets
    (signed) and accumulated, then L2-normalized. Same text -> same vector,
    always, with no downloads and no model cache. Not semantically strong, but
    it exercises the full FAISS retrieval path deterministically.
    """

    model_id = "hash-bag-of-words:v1"  # offline deterministic embedder identity

    def __init__(self, dim: int = EMBEDDING_MODEL_DIM) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.md5(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[i, bucket] += sign
        return _normalize(out)


class SentenceTransformerEmbedder:
    """The real embedding model (pinned in yt_rag.config).

    Downloads weights from Hugging Face on first use, then caches locally.
    Normalized embeddings -> inner product == cosine similarity.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        revision: str | None = EMBEDDING_MODEL_REVISION,
        **encode_kwargs: Any,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        kwargs: dict[str, Any] = {}
        if revision:
            kwargs["revision"] = revision
        self._model = SentenceTransformer(model_name, **kwargs)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())
        self.model_id = (
            f"{model_name}@{revision}" if revision else model_name
        )  # exact identity for the artifact manifest
        self._encode_kwargs = {"batch_size": 32, **encode_kwargs}

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(texts, normalize_embeddings=True, **self._encode_kwargs)
        return _normalize(vectors)
