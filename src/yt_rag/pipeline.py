"""End-to-end RAG pipeline: ingest -> chunk -> embed -> index -> retrieve -> generate."""

from __future__ import annotations

from pathlib import Path

from yt_rag.bundle import embedder_model_id
from yt_rag.bundle import load_bundle as _load_bundle
from yt_rag.bundle import save_bundle as _save_bundle
from yt_rag.chunk import Chunk, chunk_text
from yt_rag.config import ARTIFACTS_DIR, DEFAULT_TOP_K
from yt_rag.embeddings import EmbeddingProvider, HashEmbedder
from yt_rag.generation import GenerationProvider, StubProvider
from yt_rag.ingest import Transcript, fetch_transcript, load_transcript_file
from yt_rag.retriever import RetrievedChunk, Retriever
from yt_rag.vectorstore import FaissVectorStore


class RAGPipeline:
    """Real RAG over one transcript.

    The store holds ONLY chunk vectors; generation sees only the top-k
    retrieved chunks, never the full transcript.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        provider: GenerationProvider | None = None,
        top_k: int = DEFAULT_TOP_K,
        artifacts_dir: str | Path | None = None,
    ) -> None:
        self.embedder = embedder or HashEmbedder()
        self.provider = provider or StubProvider()
        self.top_k = top_k
        self.artifacts_dir = Path(artifacts_dir or ARTIFACTS_DIR)
        self.store: FaissVectorStore | None = None
        self.retriever: Retriever | None = None

    # --- ingest -------------------------------------------------------------
    def ingest_transcript(self, transcript: Transcript) -> list[Chunk]:
        """Chunk, embed, and index one transcript. Returns the chunks."""
        chunks = chunk_text(transcript.text)
        if not chunks:
            raise ValueError("transcript produced no chunks")
        vectors = self.embedder.embed([c.text for c in chunks])
        store = FaissVectorStore(dim=vectors.shape[1])
        store.add(vectors, chunks)
        self.store = store
        self.retriever = Retriever(store, self.embedder, k=self.top_k)
        return chunks

    def ingest_from_url(self, url_or_id: str) -> list[Chunk]:
        return self.ingest_transcript(fetch_transcript(url_or_id))

    def ingest_from_file(self, path: str | Path) -> list[Chunk]:
        return self.ingest_transcript(load_transcript_file(path))

    # --- index persistence ---------------------------------------------------
    def save_index(self, directory: str | Path | None = None) -> Path:
        if self.store is None:
            raise RuntimeError("nothing ingested yet")
        out = Path(directory) if directory else self.artifacts_dir / "default"
        return self.store.save(out)

    def load_index(self, directory: str | Path) -> None:
        self.store = FaissVectorStore.load(directory)
        self.retriever = Retriever(self.store, self.embedder, k=self.top_k)

    # --- artifact bundle (Stage 3) -------------------------------------------
    def save_bundle(self, directory: str | Path | None = None) -> Path:
        """Persist the index + identity manifest (see yt_rag.bundle)."""
        if self.store is None:
            raise RuntimeError("nothing ingested yet")
        out = Path(directory) if directory else self.artifacts_dir / "bundle"
        return _save_bundle(
            self.store, out, model_id=embedder_model_id(self.embedder), top_k=self.top_k
        )

    def load_bundle(self, directory: str | Path) -> dict:
        """Load an artifact bundle and validate it against this embedder.

        Raises BundleError if the bundle's recorded embedder identity or dim
        does not match this pipeline's embedder.
        """
        store, manifest = _load_bundle(
            directory,
            expected_model_id=embedder_model_id(self.embedder),
            expected_dim=self.embedder.dim,
        )
        self.store = store
        self.retriever = Retriever(store, self.embedder, k=self.top_k)
        return manifest

    # --- retrieval + generation ----------------------------------------------
    def retrieve(self, question: str) -> list[RetrievedChunk]:
        if self.retriever is None:
            raise RuntimeError("ingest or load an index before retrieving")
        return self.retriever.retrieve(question)

    def ask(self, question: str) -> dict:
        """Retrieve top-k chunks and generate an answer grounded in them."""
        retrieved = self.retrieve(question)
        answer = self.provider.generate(question, retrieved)
        return {
            "question": question,
            "answer": answer,
            "retrieved": [
                {"chunk_index": r.chunk.index, "score": r.score, "text": r.chunk.text}
                for r in retrieved
            ],
        }
