"""End-to-end pipeline tests (offline fixtures, HashEmbedder, StubProvider)."""

import pytest

from yt_rag.embeddings import HashEmbedder
from yt_rag.errors import IngestError
from yt_rag.generation import StubProvider
from yt_rag.pipeline import RAGPipeline


@pytest.fixture()
def pipe(tmp_path):
    return RAGPipeline(
        embedder=HashEmbedder(dim=384),
        provider=StubProvider(),
        artifacts_dir=tmp_path / "artifacts",
    )


def test_full_pipeline_ingest_retrieve_generate(pipe, teal_fixture):
    chunks = pipe.ingest_from_file(str(teal_fixture))
    assert len(chunks) > 5
    result = pipe.ask("four ways to use chatgpt to optimize your linkedin profile")
    assert result["question"] == "four ways to use chatgpt to optimize your linkedin profile"
    assert result["answer"]
    assert 0 < len(result["retrieved"]) <= pipe.top_k
    # grounded: answer text must come from retrieved chunks (stub echoes them)
    retrieved_text = " ".join(r["text"] for r in result["retrieved"])
    assert "stub answer" in result["answer"]
    best = result["retrieved"][0]
    assert best["text"][:40] in retrieved_text


def test_save_and_reload_index(tmp_path, teal_fixture):
    pipe = RAGPipeline(
        embedder=HashEmbedder(dim=384),
        provider=StubProvider(),
        artifacts_dir=tmp_path / "artifacts",
    )
    pipe.ingest_from_file(str(teal_fixture))
    out = pipe.save_index()
    assert (out / "index.faiss").is_file()

    fresh = RAGPipeline(
        embedder=HashEmbedder(dim=384),
        provider=StubProvider(),
        artifacts_dir=tmp_path / "artifacts",
    )
    fresh.load_index(out)
    a = pipe.retrieve("chatgpt linkedin profile")
    b = fresh.retrieve("chatgpt linkedin profile")
    assert [h.chunk.index for h in a] == [h.chunk.index for h in b]
    assert [h.score for h in a] == pytest.approx([h.score for h in b])


def test_ask_before_ingest_raises(pipe):
    with pytest.raises(RuntimeError):
        pipe.ask("anything")


def test_ingest_bad_url_raises(pipe):
    with pytest.raises(IngestError):
        pipe.ingest_from_url("definitely-not-a-youtube-url")
