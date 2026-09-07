# yt_rag — RAG over YouTube transcripts

## Motivation

Turn the old "YouTube Transcript Analyzer" into a portfolio-grade, full-lifecycle
retrieval-augmented generation (RAG) application over YouTube transcripts. The
previous app answered questions with a single LLM call over the whole transcript;
this rebuild demonstrates real retrieval: chunking, embeddings, vector search,
and top-k retrieval feeding generation.

## Method

**Not implemented yet.** This is a Stage 0 scaffold. The planned pipeline
(see ROADMAP.md, Stage 1) is:

- Transcript ingestion (youtube-transcript-api based).
- Chunking with a documented strategy.
- Embeddings (sentence-transformers, pinned version).
- Vector store (FAISS) with save/load.
- Top-k retrieval with a documented similarity metric.
- Generation step that uses retrieved context.

Evaluation (hit rate, mean reciprocal rank) comes before any tuning (Stage 2).

## Results

No results yet. No RAG functionality is implemented in Stage 0. This repository
currently contains only the package skeleton, a smoke test, and tooling
(lockfile, pytest, ruff, CI, Makefile).

## Limitations

- No ingestion, chunking, embeddings, retrieval, or generation is implemented.
- The smoke test only checks the Python version and that `yt_rag` imports.
- Requires Python 3.11 or newer.

## Operational notes

Setup on a clean machine (Linux, macOS, or Windows Git Bash):

```
git clone <repo-url>
cd youtube_app
make setup
make test
```

`make setup` creates `.venv`, upgrades pip, and installs the pinned
dependencies from `requirements-lock.txt`. `make test` runs pytest and ruff.
`make lint` additionally checks formatting. Do not commit secrets or API keys;
configuration is environment-based by policy.
