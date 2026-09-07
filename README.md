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

Stage 2 pre-tuning baselines, measured on a **14-item hand-labeled eval set**
(12 answerable items with a gold span literally present in the fixture
transcript, plus 2 items whose answer is absent from the transcript) over the
two cached fixture transcripts. Gold chunk indices are the chunks (pinned
800/150 chunker) containing the gold span verbatim. The set is small and labeled
by the project author, so read these as baselines to improve on, not benchmarks.
Full run records: `experiments/runs/`; log: `experiments/baseline_log.md`;
reproduce with `make eval` (commit `ceb5ecf`, 2026-09-07).

| Embedder | hit rate@4 | MRR@4 | Groundedness (lexical) | Notes |
| --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 (pinned) | 0.917 (11/12) | 0.653 | 0.907 (STUB) | quality baseline |
| HashEmbedder | 0.917 (11/12) | 0.660 | 0.905 (STUB) | deterministic smoke, **NOT a quality measure** |

How to read this:

- **hit rate@4** is the fraction of the 12 answerable queries whose top-4
  retrieved chunks include a gold chunk; **MRR@4** is the mean reciprocal rank
  of the first gold chunk. The two runs retrieve different chunk lists per
  query; the equal hit rates are a coincidence at this sample size.
- **Groundedness is STUB**, and both runs used `StubProvider` for generation.
  The stub is extractive — it echoes retrieved sentences — so high lexical
  overlap with retrieved chunks is expected by construction. It measures that
  answers are built from retrieved text, not that an LLM answers well.
- **Refusal on the 2 absent-answer questions is 0.0** in both runs:
  `StubProvider` cannot say "cannot find it in the transcript". Measuring
  refusal requires an LLM provider.
- **No qualitative LLM review was run**: `OPENAI_COMPATIBLE_API_KEY` was not
  set on the eval machine. `make eval` records qualitative LLM notes
  automatically when the key is present.
- **These are pre-tuning baselines.** Chunk size/overlap (800/150), top-k (4),
  similarity metric (cosine via IndexFlatIP), and the embedding model are the
  Stage 1 values, unchanged. No tuning has happened yet.

## Limitations

- **The eval set is small (14 items) and hand-labeled by the project author.**
  It covers only the two cached fixture transcripts, and 12 of those items are
  scoreable. A 0.917 hit rate@4 means one miss; the confidence interval on a
  12-item denominator is wide. Gold chunk relevance is defined as "chunk text
  contains the gold span verbatim", which is narrow.
- **The HashEmbedder CI run is not a quality signal.** It is a deterministic
  bag-of-words hash, not semantically meaningful; it exists so `make test`
  stays offline and the eval harness is exercised deterministically. Its
  baseline numbers are recorded for reproducibility only.
- **Stub groundedness is not LLM quality.** All groundedness numbers above were
  produced by `StubProvider`, an extractive stub that echoes retrieved
  passages. They show the answer is built from retrieved text, nothing more.
- **No tuning has happened yet.** Every number in Results is a pre-tuning
  baseline of the Stage 1 pipeline unchanged (800/150, k=4, cosine,
  all-MiniLM-L6-v2). No optimization informed these numbers, and none has been
  done since.
- **No qualitative LLM review yet.** The eval machine had no
  `OPENAI_COMPATIBLE_API_KEY`, so generation quality was not reviewed by a real
  LLM, and the cannot-find-it refusal rate could not be measured (the stub
  cannot refuse).
- Retrieval is exact flat search (O(n) per query) over one transcript; it is
  not scaled for many-document corpora.
- Transcripts are plain text with no speaker or timestamp structure; chunk
  boundaries can split a speaker turn.
- youtube-transcript-api depends on YouTube's transcript availability; videos
  without captions cannot be ingested.
- **Stage 5 observability is local and shallow, and its retrieval-quality
  signal is a PROXY.** Metrics are in-process counters that reset on restart;
  there is no metrics store, no Prometheus/Grafana, and no alerting. The
  "retrieval-quality proxy" (empty-result rate, top score) shows whether
  retrieval returned anything and how confident the vector search was — it is
  not answer quality. Stage 2 groundedness remains a stub and the qualitative
  LLM review is still open; nothing here relabels that.
- Requires Python 3.12 or newer (the pinned lock resolved numpy 2.5.3, which
dropped support for 3.11; `pyproject.toml` and CI were updated to match).

## Operational notes

Setup on a clean machine (Linux, macOS, or Windows Git Bash; Python **3.12+** —
the single lock resolved packages that no longer support 3.11):

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

`make eval` runs the Stage 2 evaluation harness and is **not** part of
`make test`: it may download the pinned MiniLM weights on first use (network),
and if `OPENAI_COMPATIBLE_API_KEY` is set it also records qualitative LLM
notes. It writes one JSON run record per embedder under `experiments/runs/`
against the labeled set in `experiments/eval_set.json`.

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

### Versioning (Stage 3)

- **Embedding model pin.** The retrieval core is pinned to
  `sentence-transformers/all-MiniLM-L6-v2` (384-dim) in
  `src/yt_rag/config.py` (`EMBEDDING_MODEL_NAME`, `EMBEDDING_MODEL_DIM`), with
  an unused-by-default revision hook (`EMBEDDING_MODEL_REVISION`) for a
  bit-exact Hugging Face commit pin if upstream ever re-uploads the model.
  Library versions are pinned in `requirements-lock.txt` — the single lockfile;
  there is deliberately no second lock.
- **Git tag convention.** Stage releases are tagged `stageN-vX.Y.Z` (annotated
  tags, e.g. `stage3-v0.1.0`), matching the package version in
  `pyproject.toml` at that point. Tags are local bookkeeping for this portfolio
  repo; nothing is pushed to a registry and nothing requires a tag to build or
  run.

### Artifact bundle (Stage 3)

A bundle is one directory under the gitignored `artifacts/` path containing the
FAISS index (`index.faiss`), the chunk metadata (`chunks.json`), and
`manifest.json` — identity metadata only (`model_id`, `dim`, chunk config,
top-k, package version, `created_at`, `git_commit`). No model weights are
stored anywhere in the repo: the real MiniLM weights stay in the local
Hugging Face cache, and the manifest identifies them instead. The schema lives
in `yt_rag.bundle.build_manifest`; the committed module docstring is the
reference.

Save (offline, fixture + HashEmbedder; add `--real-embedder` for the pinned
model — that variant downloads weights once):

```
.venv/Scripts/python -m yt_rag.cli --file tests/fixtures/teal_chatgpt_linkedin.txt \
    --question "how do I optimize my LinkedIn profile with ChatGPT" --save-bundle
# -> artifacts/bundle/{index.faiss, chunks.json, manifest.json}
```

A reviewer then has two reproduction paths:

1. **Reload** the bundle and query it with an embedder matching the manifest
   identity (`model_id` + `dim` are validated; a mismatch raises `BundleError`
   instead of silently returning wrong embeddings):

```
.venv/Scripts/python -c "from yt_rag.embeddings import HashEmbedder; from yt_rag.pipeline import RAGPipeline; p = RAGPipeline(embedder=HashEmbedder(dim=384)); m = p.load_bundle('artifacts/bundle'); print(m['embedder']); print(p.ask('how do I optimize my LinkedIn profile with ChatGPT')['answer'])"
```

2. **Rebuild from source**: fresh clone -> `make setup` -> re-run the ingest +
   embed command above with the same embedder; the manifest records exactly
   which model and chunk config produced the original index.

### Docker (local compose, Stage 3 — not a production deploy)

`Dockerfile` + `docker-compose.yml` package the app for **local** runs. The
image installs from `requirements-lock.txt` (the single lock) plus the package
itself; no API keys or `.env` files are baked in or mounted. The default
in-container path is offline: `HashEmbedder` + `StubProvider`, with the
committed fixture transcripts copied into the image.

```
docker compose build              # build proof; also run in CI (no registry push)
docker compose up -d              # serve on http://localhost:8000
curl http://localhost:8000/health # -> {"status":"ok"} (Docker healthcheck hits the same endpoint)
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
    -d '{"file": "tests/fixtures/teal_chatgpt_linkedin.txt", "question": "how do I optimize my LinkedIn profile with ChatGPT"}'
docker compose down
```

Offline CLI smoke inside the container (no network, no keys):

```
docker compose run --rm yt-rag python -m yt_rag.cli \
    --file tests/fixtures/teal_chatgpt_linkedin.txt \
    --question "how do I optimize my LinkedIn profile with ChatGPT"
```

**Offline vs networked paths.** The compose health/smoke path above requires
no network and no API keys. The networked options are opt-in and documented,
not exercised by any default command: running the pipeline with
`--real-embedder` downloads the pinned MiniLM weights once (then cached), and
generation via `--llm --llm-model <name>` needs
`OPENAI_COMPATIBLE_API_KEY` plus an OpenAI-compatible endpoint. Pass secrets at
run time (e.g. `docker compose run --rm -e OPENAI_COMPATIBLE_API_KEY yt-rag ...`),
never in the image.

**Image size note.** `requirements-lock.txt` is a universal lock that includes
torch (a dependency of sentence-transformers, needed for the real-model path),
so the Linux image is several GB. This is the honest cost of one lockfile
covering Windows dev and Linux containers; a CPU-only or slim image would need
a second lock, which Stage 3 explicitly avoids.

### API (documented in Stage 4; the serving layer itself is unchanged since Stage 1)

The serving layer is `yt_rag.app` (FastAPI, one process, no auth, no sessions —
single-user local serving). Start it with `uvicorn yt_rag.app:app`, or via the
compose service below. FastAPI generates the machine-readable schema and
interactive docs from the same code that serves the endpoints: **`/docs`**
(Swagger UI) and `/redoc`. The schemas below are documentation of what is
there, not a second specification.

**`GET /health`** — liveness probe. No request body. Returns 200
`{"status": "ok"}`. The Docker healthcheck hits this endpoint.

**`GET /metrics`** (Stage 5) — in-process observability counters as JSON; see
"Observability (Stage 5)" below. The counters live in the serving process only
and reset on restart; `/health` and `/metrics` themselves are logged but not
counted.

**`POST /chat`** — runs the whole pipeline per request: ingest -> chunk ->
embed -> index -> retrieve top-k -> generate from the retrieved chunks only
(default in-container path: `HashEmbedder` + `StubProvider`, i.e. the answer is
a deterministic extractive stub, not an LLM).

Request (`application/json`):

```json
{
  "url": "https://www.youtube.com/watch?v=<11-char-id>",
  "file": "tests/fixtures/teal_chatgpt_linkedin.txt",
  "question": "how do I optimize my LinkedIn profile with ChatGPT"
}
```

- `url` (optional): a YouTube URL or bare 11-char video ID (live fetch, network).
- `file` (optional): path to a cached transcript `.txt`, relative to the
  container workdir `/app`.
- Exactly one of `url` / `file` is required; giving neither returns 422.
- `question` (required): the query.

Response 200 (`ChatResponse`):

```json
{
  "question": "how do I optimize my LinkedIn profile with ChatGPT",
  "answer": "[stub answer, grounded in 4 retrieved chunk(s)]\nRelevant transcript passage (chunk 3):\n...",
  "retrieved": [
    {"chunk_index": 3, "score": 0.318, "text": "chunk text..."}
  ]
}
```

`retrieved` holds up to 4 chunks (top-k), best first, `score` = cosine
similarity between the question vector and the chunk vector.

Status codes:

- **200** — an answer was produced from retrieved chunks.
- **422** — request validation (malformed body, missing `question`) or
  ingestion failure (`IngestError`: neither `url` nor `file` given,
  unrecognizable URL, video without captions, network fetch failure, missing
  or empty transcript file).
- **400** — pipeline value/state errors (`ValueError`/`RuntimeError`, e.g. a
  transcript that chunks down to zero chunks).
- **500** — unexpected server error. The default `StubProvider` cannot fail;
  if a real provider is injected and the upstream call fails, that surfaces as
  500 (`GenerationError` is deliberately not mapped to a 4xx: the request was
  valid, the server-side dependency failed).

### Observability (Stage 5 — local, in-process; NOT production monitoring)

Stage 5 adds two additive signals to the same compose-local service. No new
dependencies (stdlib `logging` + FastAPI), no extra containers, no Grafana.

**Structured JSON logs (stdout, one object per line).** Every `/chat` request
emits one line; `/health` and `/metrics` emit minimal lines. Example:

```json
{"ts": "2026-09-07T20:27:42.224+00:00", "level": "INFO", "request_id": "7ac06d1b43f5",
 "endpoint": "/chat", "status": 200, "total_ms": 2.86, "retrieval_ms": 0.06,
 "generation_ms": 0.0, "error_class": null, "question_chars": 8,
 "retrieved_count": 4, "top_score": 0.0, "empty_result": false}
```

Fields: `request_id` (random, opaque), `endpoint`, `status`, `total_ms`,
`retrieval_ms` / `generation_ms` (measured separately around retrieve vs
generate in `RAGPipeline.ask`), `error_class` (exception type name, e.g.
`IngestError` for a 422), and the retrieval proxy fields `retrieved_count`,
`top_score` (best cosine score, rounded), `empty_result`. **No secrets and no
content in logs**: API keys are never read by the logging path, and the
question, answer, and retrieved text are never logged — only
counts/scores/ids/timings (`question_chars` is a length, not the question).

**`GET /metrics` (JSON, in-process).** Returns request count, error count /
rate / classes, latency summaries (`count`, `mean`, `p50`, `p95`, `max` for
total, retrieval, and generation), and the retrieval-quality proxy. Counters
are plain process memory: **they reset on restart**, cover `/chat` traffic
only, and are not shared across workers. There is deliberately no metrics
store, scrape interval, retention, or alerting — this documents what a
reviewer can see by curling the running container, nothing more.

**Retrieval-quality PROXY (label it exactly that).** `empty_result_count` /
`empty_result_rate` (requests where retrieval returned 0 chunks) and
`mean_top_score` (mean best cosine score). These are **proxies for retrieval
health, not quality measures**: a nonzero retrieval with a low top score says
the vector search found nothing similar; it says nothing about answer
correctness. Stage 2's groundedness metric is a STUB and the qualitative LLM
review is open — Stage 5 does not change or relabel either.

**What could degrade, and the signal that would show it:**

| Degradation | Signal in logs / metrics |
| --- | --- |
| Transcript format / API changes (youtube-transcript-api breaks, captions removed) | `/chat` 422 with `error_class: "IngestError"` rising in `error_classes` and `error_rate` |
| Embedding-model drift (pinned model re-uploaded upstream, local cache vs fresh download differ) | `top_score` / `mean_top_score` sliding down with no code change; no hard error |
| Index staleness (index built with a different chunk config or embedder than the serving pipeline's current config) | low `top_score` across requests, or a `BundleError`/`ValueError` on load (`error_class` 400s); bundle identity validation is the stronger guard (Stage 3) |
| Empty retrieval | `empty_result: true` lines; `empty_result_count` / `empty_result_rate` in `/metrics` |
| Stub vs real LLM behavior differences | stub cannot fail or refuse; a real provider failure surfaces as `error_class: "GenerationError"` with a 500 — watch `error_rate` and `generation_ms` |

**Compose verification (branch `stage5`).** Because `src/yt_rag/app.py`
changed, the image was rebuilt (`docker compose build`, ~15 s with cached
layers) and the service exercised with real curls: `/health` 200, `/chat` 200
(fixture-backed), `/metrics` JSON (`request_count`, latency summaries, proxy),
a no-source `/chat` returning 422 with `error_class: "IngestError"` in both the
log line and `error_classes`, then `docker compose down`.

**Scope honesty:** this is observability of a local Docker Compose service on
committed fixture transcripts. It is not production monitoring: no dashboards,
no alerting, no multi-process aggregation, no history across restarts. The
optional Prometheus/Grafana dashboard roadmap item is left unchecked — nothing
was stood up.

### Deploy target decision (Stage 4)

**Local Docker Compose is the deploy target** — accepted as the minimum: one
command serves the API on `http://localhost:8000`, which is everything a
reviewer cloning this repo needs. A public endpoint (ngrok, Azure, AWS, or any
paid hosting) was **declined**: this is a portfolio project, not a production
service; the image is several GB (torch in the single lock), and hosting it
publicly would add cost and attack surface for no reviewer benefit. The honest
consequence: **no TLS, no auth, no multi-user serving, no registry push —
nothing in this repo is publicly reachable.**

### Environment variables (all optional — no hardcoded secrets)

The default compose path runs fully offline (`HashEmbedder` + `StubProvider` +
committed fixtures): `docker compose up` works with **no key and no network**.
The code reads exactly three environment variables; none is required:

| Variable | Read by | When unset | Purpose |
| --- | --- | --- | --- |
| `OPENAI_COMPATIBLE_API_KEY` | `OpenAICompatibleProvider` (opt-in real-LLM path) and `make eval` qualitative notes | provider raises `GenerationError`; eval skips LLM notes | API key for the OpenAI-compatible endpoint. The only secret the code can read; passed at run time (`docker compose run --rm -e OPENAI_COMPATIBLE_API_KEY yt-rag ...`), never committed or baked into the image. |
| `YT_RAG_ARTIFACTS_DIR` | `config.py` | `<repo>/artifacts` | Where FAISS indexes and bundles are written. |
| `YT_RAG_EVAL_LLM_MODEL` | `make eval` qualitative notes | `openai/gpt-4o-mini` | Model used for the optional LLM notes during eval. |

The real-LLM **base URL and model name are not environment variables**: they
are the `--llm-base-url` / `--llm-model` CLI flags and
`OpenAICompatibleProvider` constructor arguments (default base URL
`https://openrouter.ai/api/v1`). The remaining `config.py` constants (the
embedding model pin, chunk size/overlap, top-k, `EMBEDDING_MODEL_REVISION`)
were audited for Stage 4 and deliberately left in code: they are reproducibility
pins, not secrets or host-specific settings, and `config.py` contains no
hardcoded credentials.

### Deployment runbook (local Docker Compose, verified on branch `stage4`)

```
git clone <repo-url> && cd youtube_app
make setup && make test          # optional; offline, no Docker needed
docker compose up -d             # builds yt-rag:stage3 if absent, serves http://localhost:8000
curl -s http://localhost:8000/health
# {"status":"ok"}

curl -s -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"file": "tests/fixtures/teal_chatgpt_linkedin.txt", "question": "how do I optimize my LinkedIn profile with ChatGPT"}'
# -> ChatResponse JSON (answer + retrieved chunks); see "API" above

docker compose down
```

The fixture path is inside the container (`WORKDIR /app`, fixtures copied by
the Dockerfile), not a host path. Windows Git Bash notes:

- Quote the JSON body in single quotes with plain double quotes inside, exactly
  as above; Git Bash passes it through without Windows `\"` escaping.
- If a proxy environment variable intercepts localhost calls, add
  `--noproxy '*'` to the curl commands.
- Use forward slashes in the `"file"` value; it is a container path.

After `docker compose down` the container and the published port are gone; the
image `yt-rag:stage3` stays cached locally. Nothing is pushed anywhere.
