"""yt_rag: retrieval-augmented generation over YouTube transcripts.

Real RAG pipeline (Stage 1): transcript ingestion, chunking with overlap,
embeddings (pinned sentence-transformers model; deterministic offline
HashEmbedder for tests), FAISS vector store with save/load, top-k retrieval
(inner product on normalized vectors = cosine similarity), and generation
restricted to retrieved chunks via a swappable provider.
"""

__version__ = "0.1.0"
