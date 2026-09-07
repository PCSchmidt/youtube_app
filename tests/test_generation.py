"""Generation tests: stub determinism, grounded-only context, provider errors."""

import json

import httpx
import pytest

from yt_rag.chunk import Chunk
from yt_rag.errors import GenerationError
from yt_rag.generation import (
    OpenAICompatibleProvider,
    StubProvider,
    build_context_prompt,
)
from yt_rag.retriever import RetrievedChunk


def make_retrieved():
    chunks = [
        Chunk(
            text="Sentence one about ChatGPT. Sentence two about LinkedIn.",
            index=0,
            start_char=0,
            end_char=60,
        ),
        Chunk(text="Another passage entirely.", index=3, start_char=100, end_char=125),
    ]
    return [RetrievedChunk(score=0.9, chunk=chunks[0]), RetrievedChunk(score=0.5, chunk=chunks[1])]


def test_context_prompt_contains_only_retrieved_chunks():
    prompt = build_context_prompt(make_retrieved())
    assert "Sentence one about ChatGPT" in prompt
    assert "Another passage entirely" in prompt
    assert "[Chunk 0]" in prompt and "[Chunk 3]" in prompt


def test_context_prompt_empty_retrieved_raises():
    with pytest.raises(GenerationError):
        build_context_prompt([])


def test_stub_provider_is_deterministic_and_grounded():
    stub = StubProvider()
    a = stub.generate("what is this about", make_retrieved())
    b = stub.generate("what is this about", make_retrieved())
    assert a == b
    # grounded: the echoed passage must come from the top retrieved chunk
    assert "Sentence one about ChatGPT" in a
    assert "Sentence two about LinkedIn" in a


def test_stub_ignores_non_stub_providers_contract():
    # the stub must not need anything beyond the retrieved chunks
    stub = StubProvider()
    assert isinstance(stub.generate("q", make_retrieved()), str)


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    provider = OpenAICompatibleProvider(model="test-model", base_url="http://unused")
    with pytest.raises(GenerationError, match="missing API key"):
        provider.generate("q", make_retrieved())


def test_openai_provider_success_via_mock_transport(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode())
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "grounded answer"}}]},
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")
    provider = OpenAICompatibleProvider(model="test-model", base_url="http://fake.local/v1")
    provider._client = httpx.Client(transport=transport)

    answer = provider.generate("what is this about", make_retrieved())
    assert answer == "grounded answer"
    assert captured["auth"] == "Bearer test-key"
    user_msg = captured["payload"]["messages"][-1]["content"]
    assert "Sentence one about ChatGPT" in user_msg
    assert "Another passage entirely" in user_msg


def test_openai_provider_http_error_wrapped(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")
    provider = OpenAICompatibleProvider(model="m", base_url="http://fake.local/v1")
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(GenerationError, match="failed"):
        provider.generate("q", make_retrieved())
