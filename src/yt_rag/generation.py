"""Generation providers.

The prompt sent to any provider contains ONLY retrieved transcript chunks —
never the full transcript. Two implementations:

- :class:`StubProvider`: deterministic, offline extractive answer used by the
  default test suite (no API keys, no network).
- :class:`OpenAICompatibleProvider`: talks to any OpenAI-compatible
  /chat/completions endpoint (OpenAI, OpenRouter, vLLM, Ollama, ...) via
  httpx, so the real LLM is swappable by configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import httpx

from yt_rag.errors import GenerationError
from yt_rag.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You answer questions about a YouTube transcript using ONLY the provided "
    "transcript excerpts. If the excerpts do not contain the answer, say you "
    "cannot find it in the transcript. Do not use outside knowledge."
)


def build_context_prompt(retrieved: list[RetrievedChunk]) -> str:
    """Grounded context: numbered retrieved chunks and nothing else."""
    if not retrieved:
        raise GenerationError("no retrieved chunks to ground an answer on")
    blocks = [f"[Chunk {r.chunk.index}]\n{r.chunk.text}" for r in retrieved]
    return "\n\n---\n\n".join(blocks)


class GenerationProvider(Protocol):
    """Anything that answers a question given grounded retrieved chunks."""

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> str: ...


class StubProvider:
    """Deterministic offline answer built from the top retrieved chunk.

    Not an LLM. It echoes the most relevant passages so the grounded pipeline
    can be tested end to end without keys or network. Real usage swaps in an
    OpenAICompatibleProvider.
    """

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        best = retrieved[0]
        sentences = [
            s.strip() for s in best.chunk.text.replace(". ", ".\x00").split("\x00") if s.strip()
        ]
        # Take up to three sentences nearest the start of the best chunk as a
        # crude deterministic "answer", clearly labeled as extractive.
        answer = " ".join(sentences[:3]) if sentences else best.chunk.text
        return (
            f"[stub answer, grounded in {len(retrieved)} retrieved chunk(s)]\n"
            f"Relevant transcript passage (chunk {best.chunk.index}):\n{answer}"
        )


@dataclass
class OpenAICompatibleProvider:
    """Chat-completions client for any OpenAI-compatible endpoint."""

    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENAI_COMPATIBLE_API_KEY"
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    _client: httpx.Client | None = None

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        context = build_context_prompt(retrieved)
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise GenerationError(
                f"missing API key: set {self.api_key_env} to use {type(self).__name__}"
            )
        client = self._client or httpx.Client(timeout=self.timeout_seconds)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Transcript excerpts:\n\n{context}\n\nQuestion: {question}",
                },
            ],
        }
        try:
            response = client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            return str(body["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise GenerationError(f"generation provider failed: {exc}") from exc
        finally:
            if self._client is None:
                client.close()
