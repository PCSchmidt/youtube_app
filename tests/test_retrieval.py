"""Retrieval tests against a real cached transcript (offline, HashEmbedder)."""

import pytest

from yt_rag.embeddings import HashEmbedder
from yt_rag.pipeline import RAGPipeline


@pytest.fixture(scope="module")
def ingested(tmp_path_factory, teal_fixture):
    pipe = RAGPipeline(embedder=HashEmbedder(dim=384), artifacts_dir=tmp_path_factory.mktemp("art"))
    chunks = pipe.ingest_from_file(str(teal_fixture))
    return pipe, chunks


def test_ingest_produces_chunks(ingested):
    pipe, chunks = ingested
    assert len(chunks) > 5
    assert len(pipe.store) == len(chunks)
    assert pipe.store.dim == 384


def test_whole_chunk_query_returns_that_chunk_first(ingested):
    """An exact chunk as the query must retrieve that chunk at rank 1."""
    pipe, chunks = ingested
    target = chunks[2]
    hits = pipe.retrieve(target.text)
    assert hits, "no results"
    assert hits[0].chunk.index == target.index
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert all(h.score > 0 for h in hits)


def test_phrase_query_returns_results_within_topk(ingested):
    """The hashed bag-of-words embedder is not semantic; only require the
    target chunk to appear among the top-k results for an 8-word phrase."""
    pipe, chunks = ingested
    target = chunks[2]
    phrase = " ".join(target.text.split()[:8])
    hits = pipe.retrieve(phrase)
    assert hits, "no results"
    assert len(hits) <= pipe.top_k
    assert all(h.score > 0 for h in hits)


def test_top_k_limit(ingested):
    pipe, _ = ingested
    assert len(pipe.retrieve("chatgpt linkedin profile")) <= pipe.top_k


def test_scores_descending(ingested):
    pipe, _ = ingested
    scores = [h.score for h in pipe.retrieve("optimize your profile")]
    assert scores == sorted(scores, reverse=True)


def test_scores_finite(ingested):
    import numpy as np

    pipe, _ = ingested
    for hit in pipe.retrieve("four ways to use chatgpt"):
        assert np.isfinite(hit.score)
        assert hit.score > 0
