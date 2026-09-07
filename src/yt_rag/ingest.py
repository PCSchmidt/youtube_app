"""Transcript ingestion.

Fetches transcripts from YouTube via youtube-transcript-api, with error
handling mapped to a single IngestError type so callers never see raw library
exceptions. Offline fixture loading is provided for tests and demos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from yt_rag.errors import IngestError

_VIDEO_ID_PATTERNS = (
    re.compile(r"(?:v=|/videos/|embed/|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})"),
    re.compile(r"^([A-Za-z0-9_-]{11})$"),
)


@dataclass(frozen=True)
class Transcript:
    """A fetched transcript and where it came from."""

    video_id: str
    text: str
    source: str  # "youtube" or a file path


def extract_video_id(url_or_id: str) -> str:
    """Extract an 11-char YouTube video ID from a URL or return a bare ID.

    Raises IngestError for anything else.
    """
    candidate = (url_or_id or "").strip()
    for pattern in _VIDEO_ID_PATTERNS:
        m = pattern.search(candidate)
        if m:
            return m.group(1)
    raise IngestError(f"Not a recognizable YouTube video ID or URL: {url_or_id!r}")


def _fetch_transcript_api(video_id: str) -> str:
    """Thin wrapper so tests can monkeypatch a single seam.

    The real implementation calls youtube-transcript-api. Any failure it
    raises (no captions, disabled transcripts, network errors, HTTP errors)
    propagates here and is mapped to IngestError by fetch_transcript.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    fetched = YouTubeTranscriptApi().fetch(video_id)
    return " ".join(part.text for part in fetched)


def fetch_transcript(url_or_id: str) -> Transcript:
    """Fetch a transcript from YouTube. Raises IngestError on any failure."""
    video_id = extract_video_id(url_or_id)
    try:
        text = _fetch_transcript_api(video_id)
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(f"Transcript fetch failed for {video_id}: {exc}") from exc
    if not text.strip():
        raise IngestError(f"Transcript for {video_id} is empty")
    return Transcript(video_id=video_id, text=text, source="youtube")


def load_transcript_file(path: str | Path) -> Transcript:
    """Load a cached plain-text transcript (offline fixture/demo path)."""
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"Transcript file not found: {p}")
    text = p.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise IngestError(f"Transcript file is empty: {p}")
    # Use the filename stem as a stable pseudo video ID for artifact layout.
    return Transcript(video_id=p.stem, text=text, source=str(p))
