"""Chunking tests: size bounds, overlap continuity, word boundaries."""

import itertools

import pytest

from yt_rag.chunk import chunk_text

LONG = " ".join(f"word{i}" for i in range(4000))


def test_chunks_respect_size_and_overlap():
    chunks = chunk_text(LONG, chunk_size=800, overlap=150)
    assert len(chunks) > 3
    for c in chunks:
        assert len(c.text) <= 800
        assert c.text


def test_overlap_is_shared_between_neighbors():
    chunks = chunk_text(LONG, chunk_size=800, overlap=150)
    for prev, nxt in itertools.pairwise(chunks):
        assert (
            prev.text[-150:] == nxt.text[:150]
            or prev.text.endswith(nxt.text[:150])
            or prev.text[-100:] in nxt.text[:300]
        )


def test_no_mid_word_split():
    for c in chunk_text(LONG, chunk_size=800, overlap=150):
        assert c.text == c.text.strip()
        # every token must be an original whole word
        for tok in c.text.split():
            assert f" {tok} " in f" {LONG} "


def test_short_text_single_chunk():
    chunks = chunk_text("just one short sentence", chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].text == "just one short sentence"


def test_empty_text_no_chunks():
    assert chunk_text("   ") == []


def test_bad_params():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_indices_are_sequential():
    chunks = chunk_text(LONG, chunk_size=500, overlap=100)
    assert [c.index for c in chunks] == list(range(len(chunks)))
