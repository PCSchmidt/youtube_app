"""Command-line entry point for one end-to-end query.

Offline by default (HashEmbedder + StubProvider, no downloads, no keys):

    python -m yt_rag.cli --file tests/fixtures/teal_chatgpt_linkedin.txt \
        --question "how do I optimize my LinkedIn profile with ChatGPT"

Real model path (downloads pinned weights on first use, then caches locally):

    python -m yt_rag.cli --file <transcript.txt> --question "..." --real-embedder

Real generation (requires OPENAI_COMPATIBLE_API_KEY and an OpenAI-compatible
endpoint):

    python -m yt_rag.cli --file <transcript.txt> --question "..." \
        --llm --llm-model <model-name> --llm-base-url <url>
"""

from __future__ import annotations

import argparse
import json
import sys

from yt_rag.config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yt_rag", description="RAG over a YouTube transcript")
    parser.add_argument("--url", help="YouTube URL or video ID (network)")
    parser.add_argument("--file", help="path to a cached transcript .txt (offline)")
    parser.add_argument("--question", required=True, help="the query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--real-embedder",
        action="store_true",
        help=f"use the pinned model ({EMBEDDING_MODEL_NAME}) instead of the offline hash embedder",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="use an OpenAI-compatible LLM instead of the offline stub",
    )
    parser.add_argument("--llm-model", default=None, help="model name for --llm")
    parser.add_argument("--llm-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--save-index", action="store_true", help="persist the FAISS index under artifacts/"
    )
    parser.add_argument(
        "--save-bundle",
        action="store_true",
        help="persist the FAISS index + identity manifest under artifacts/bundle/",
    )
    args = parser.parse_args(argv)

    from yt_rag.embeddings import HashEmbedder
    from yt_rag.generation import OpenAICompatibleProvider, StubProvider
    from yt_rag.pipeline import RAGPipeline

    if not (args.url or args.file):
        parser.error("provide either --url or --file")

    if args.llm:
        if not args.llm_model:
            parser.error("--llm requires --llm-model")
        provider = OpenAICompatibleProvider(model=args.llm_model, base_url=args.llm_base_url)
    else:
        provider = StubProvider()

    embedder = None
    if args.real_embedder:
        from yt_rag.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()  # downloads on first use
    else:
        embedder = HashEmbedder()

    pipeline = RAGPipeline(embedder=embedder, provider=provider, top_k=args.top_k)
    if args.file:
        pipeline.ingest_from_file(args.file)
    else:
        pipeline.ingest_from_url(args.url)
    if args.save_index:
        path = pipeline.save_index()
        print(f"index saved to {path}", file=sys.stderr)
    if args.save_bundle:
        path = pipeline.save_bundle()
        print(f"artifact bundle saved to {path}", file=sys.stderr)
    result = pipeline.ask(args.question)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
