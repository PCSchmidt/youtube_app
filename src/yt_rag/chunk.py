"""Chunking with overlap.

Sliding window over the transcript on word boundaries. 800-char windows with
150-char overlap: large enough to hold several complete sentences (inside the
pinned embedding model's 256-token window), small enough that a retrieved chunk
is a focused passage; the overlap ensures no sentence spanning a window edge is
silently split in half and lost from every chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

from yt_rag.config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of the transcript, with provenance offsets."""

    text: str
    index: int
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """Split text into overlapping chunks on word boundaries.

    Chunks are at most ``chunk_size`` characters and consecutive chunks share
    ``overlap`` characters. Never splits mid-word.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Back up to the last word boundary so we never split mid-word.
            boundary = text.rfind(" ", start + overlap, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, index=len(chunks), start_char=start, end_char=end))
        if end >= n:
            break
        start = end if overlap == 0 else max(end - overlap, start + 1)
        if text[start] != " ":
            # Snap the overlap start forward to the next word boundary so no
            # chunk ever begins mid-word (overlap may be slightly shorter).
            nxt = text.find(" ", start)
            start = nxt + 1 if nxt != -1 else n
    return chunks
