# yt_rag — RAG over YouTube transcripts

## Motivation

Turn the old "YouTube Transcript Analyzer" into a portfolio-grade, full-lifecycle
retrieval-augmented generation (RAG) application over YouTube transcripts. The
previous app answered questions with a single LLM call over the whole transcript;
this rebuild demonstrates real retrieval: chunking, embeddings, vector search,
and top-k retrieval feeding generation.

## Method

Stage 1 implements a real RAG pipeline: **ingest -> chunk -> embed -> index ->
retrieve top-k -> generate from retrieved context only**. Generation never sees
the full transcript; it receives only the retrieved chunks.

### Stack decision (Stage 0)

Python 3.11+ with sentence-transformers + faiss-cpu for embeddings and vector
search, youtube-transcript-api for ingestion, FastAPI for serving, and a
swappable OpenAI-compatible provider for generation. Why:

- **Boring and recognized.** FAISS, sentence-transformers, and FastAPI are the
  defaults a hiring manager will have seen in production RAG systems.
- **Local and free.** Embeddings run on CPU with no paid API; no vendor lock-in
  for the retrieval core.
- **Reproducible.** The embedding model is pinned by name in
  `src/yt_rag/config.py`; the library versions are pinned in
  `requirements-lock.txt` (resolved for Windows and Linux).

### Pipeline

1. **Ingestion** (`yt_rag.ingest`): fetches a transcript from a YouTube URL via
   youtube-transcript-api. Bad URLs, missing captions, empty transcripts, and
   network failures all map to one `IngestError` type with a clear message.
2. **Chunking** (`yt_rag.chunk`): sliding window of **800 characters with 150
   characters of overlap**, cut on word boundaries. Rationale: 800 chars is
   roughly 150-200 tokens, so a chunk holds several complete sentences while
   staying inside the pinned embedding model's 256-token window, and a
   retrieved chunk is small enough to pack several into a prompt. The 150-char
   overlap (~19% of a chunk) means a sentence that straddles a window edge is
   never lost: it appears whole in at least one chunk.
3. **Embeddings** (`yt_rag.embeddings`): the pinned model
   `sentence-transformers/all-MiniLM-L6-v2` (384-dim, normalized). All
   embeddings are L2-normalized.
4. **Vector store** (`yt_rag.vectorstore`): a FAISS `IndexFlatIP` (exact inner
   product) wrapped with chunk metadata, with save/load to the gitignored
   `artifacts/` directory. Flat search is exact and appropriate at
   transcript-scale corpus sizes; swapping in an ANN index later changes
   nothing outside this module.
5. **Retrieval** (`yt_rag.retriever`): **inner product on L2-normalized
   vectors, which is exactly cosine similarity**. Default top-k is 4.
6. **Generation** (`yt_rag.generation`): the prompt is built only from the
   numbered retrieved chunks. Two providers implement one interface:
   `StubProvider` (deterministic offline extractive answer, used by tests) and
   `OpenAICompatibleProvider` (any OpenAI-compatible `/chat/completions`
   endpoint via httpx, so the real LLM is swappable by configuration).
7. **Serving** (`yt_rag.app`): a minimal FastAPI `POST /chat` endpoint that
   runs the whole pipeline per request; `GET /health` for liveness.

### Offline-by-default tests

The first use of a sentence-transformers model downloads weights, which would
make `make test` need network and be non-deterministic. The default test path
therefore uses a deterministic hash-based `HashEmbedder` (same 384-dim shape,
L2-normalized, exactly reproducible) plus the `StubProvider`. The real pinned
model is wired into the production path (CLI `--real-embedder`, or instantiate
`SentenceTransformerEmbedder`) and is exercised by the developer manually; the
test suite never downloads it. Cached transcript `.txt` fixtures (copied from
the read-only `legacy/` reference) exercise ingestion-file, chunking,
embedding, indexing, retrieval, and generation offline.

## Results

Not measured yet. The Stage 1 pipeline is functionally complete and verified by
58 offline tests plus a manual end-to-end run of the real model path, but no
retrieval- or generation-quality numbers exist yet.

Evaluation (hit rate, mean reciprocal rank, faithfulness) is **Stage 2** of the
roadmap and comes before any tuning. No quality claims should be inferred from
this section.

## Limitations

- **No evaluation yet.** Retrieval metrics (hit rate, MRR) and generation
  metrics (faithfulness/groundedness) are Stage 2. Nothing in this repo has
  been measured for quality.
- The default test path uses `HashEmbedder`, a hashed bag-of-words stand-in.
  It is deterministic but **not semantically meaningful**; it exists so tests
  stay offline. Retrieval quality with `HashEmbedder` says nothing about
  quality with the real model.
- `StubProvider` is not an LLM. It echoes the top retrieved passage so the
  grounded pipeline can be exercised without keys or network.
- Retrieval is exact flat search (O(n) per query) over one transcript; it is
  not scaled for many-document corpora.
- Transcripts are plain text with no speaker or timestamp structure; chunk
  boundaries can split a speaker turn.
- youtube-transcript-api depends on YouTube's transcript availability; videos
  without captions cannot be ingested.
- Requires Python 3.11 or newer.

## Operational notes

Setup on a clean machine (Linux, macOS, or Windows Git Bash):

```
git clone <repo-url>
cd youtube_app
make setup
make test
```

`make setup` creates `.venv`, upgrades pip, installs the pinned dependencies
from `requirements-lock.txt`, and installs the package editable. `make test`
runs pytest and ruff. `make lint` additionally checks formatting. Do not commit
secrets or API keys; configuration is environment-based by policy.

One offline query end to end (no keys, no network, uses a fixture transcript):

```
.venv/Scripts/python -m yt_rag.cli --file tests/fixtures/teal_chatgpt_linkedin.txt \
    --question "how do I optimize my LinkedIn profile with ChatGPT"
```

(On Linux/macOS: `.venv/bin/python -m yt_rag.cli ...`.)

Add `--save-index` to persist the FAISS index under `artifacts/` (gitignored),
`--real-embedder` to use the pinned sentence-transformers model instead of the
offline hash embedder (downloads model weights on first use, then caches them
locally; requires network once), and `--llm --llm-model <name>` to generate
with an OpenAI-compatible endpoint (requires `OPENAI_COMPATIBLE_API_KEY` and
network). The same flow is available over HTTP: start the API with
`uvicorn yt_rag.app:app` and `POST /chat` with `{"file": "...", "question":
"..."}` (or `"url"` for a live YouTube fetch).
