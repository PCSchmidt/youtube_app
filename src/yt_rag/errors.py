"""Shared exceptions for the yt_rag pipeline."""


class IngestError(Exception):
    """Raised when transcript ingestion fails (bad URL, no captions, network)."""


class GenerationError(Exception):
    """Raised when the generation provider fails to produce an answer."""


class IndexStateError(Exception):
    """Raised when a FAISS index is used before it is built or loaded."""
