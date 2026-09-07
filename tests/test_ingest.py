"""Ingestion tests: video-ID parsing, file loading, and error handling.

The youtube-transcript-api seam (ingest._fetch_transcript_api) is monkeypatched,
so nothing touches the network; the error paths still exercise real wrapper logic.
"""

import pytest

from yt_rag import ingest
from yt_rag.errors import IngestError
from yt_rag.ingest import extract_video_id, fetch_transcript, load_transcript_file


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("https://www.youtube.com/watch?v=-1nV-T9CfXE", "-1nV-T9CfXE"),
        ("https://youtu.be/V9_RzjqCXP8", "V9_RzjqCXP8"),
        ("https://www.youtube.com/embed/V9_RzjqCXP8", "V9_RzjqCXP8"),
        ("https://www.youtube.com/shorts/-1nV-T9CfXE", "-1nV-T9CfXE"),
        ("V9_RzjqCXP8", "V9_RzjqCXP8"),
    ],
)
def test_extract_video_id(given, expected):
    assert extract_video_id(given) == expected


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "not a url", "https://example.com/watch?v=short", "abcd"],
)
def test_extract_video_id_rejects_bad_input(bad):
    with pytest.raises(IngestError):
        extract_video_id(bad)


def test_load_transcript_file(teal_fixture):
    transcript = load_transcript_file(teal_fixture)
    assert transcript.video_id == "teal_chatgpt_linkedin"
    assert len(transcript.text) > 1000
    assert transcript.source == str(teal_fixture)


def test_load_transcript_file_missing(tmp_path):
    with pytest.raises(IngestError, match="not found"):
        load_transcript_file(tmp_path / "nope.txt")


def test_load_transcript_file_empty(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(IngestError, match="empty"):
        load_transcript_file(empty)


@pytest.mark.parametrize("message", ["No transcripts found", "connection timed out"])
def test_fetch_transcript_wraps_errors(monkeypatch, message):
    def fake_fetch(vid):
        raise RuntimeError(message)

    monkeypatch.setattr(ingest, "_fetch_transcript_api", fake_fetch)
    with pytest.raises(IngestError, match=message):
        fetch_transcript("https://youtu.be/-1nV-T9CfXE")


def test_fetch_transcript_empty_text_raises(monkeypatch):
    monkeypatch.setattr(ingest, "_fetch_transcript_api", lambda vid: "   ")
    with pytest.raises(IngestError, match="empty"):
        fetch_transcript("-1nV-T9CfXE")


def test_fetch_transcript_ok(monkeypatch):
    monkeypatch.setattr(ingest, "_fetch_transcript_api", lambda vid: "hello world transcript body")
    t = fetch_transcript("https://www.youtube.com/watch?v=-1nV-T9CfXE")
    assert t.video_id == "-1nV-T9CfXE"
    assert t.source == "youtube"
    assert "transcript body" in t.text
