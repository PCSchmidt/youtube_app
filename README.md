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
test suite never downloads it. Cached transcript `.txt` fixtures (salvaged
from the pre-refactor app during the Stage 0 inventory) exercise ingestion-file, chunking,
embedding, indexing, retrieval, and generation offline.

## Results

All numbers below come from the same two offline eval runs (commit `a8358d5`,
2026-09-08; run records in `experiments/runs/`, log:
`experiments/baseline_log.md`, reproduce with `make eval`), measured on a
**14-item hand-labeled eval set** (12 answerable items with a gold span
literally present in the fixture transcript, plus 2 items whose answer is
absent from the transcript) over the two cached fixture transcripts. The set
is small and labeled by the project author, so read these as baselines to
improve on, not benchmarks.

### Retrieval quality (hit rate@4 / MRR@4)

| Embedder                  | hit rate@4    | MRR@4 |
| ------------------------- | ------------- | ----- |
| all-MiniLM-L6-v2 (pinned) | 0.917 (11/12) | 0.653 |
| HashEmbedder              | 0.917 (11/12) | 0.660 |

- **hit rate@4** is the fraction of the 12 answerable queries whose top-4
  retrieved chunks include a gold chunk; **MRR@4** is the mean reciprocal rank
  of the first gold chunk. The two runs retrieve different chunk lists per
  query; the equal hit rates are a coincidence at this sample size.
- **These are pre-tuning baselines.** Chunk size/overlap (800/150), top-k (4),
  similarity metric (cosine via IndexFlatIP), and the embedding model are the
  Stage 1 values, unchanged. No tuning has happened yet.

### Groundedness evaluation (deterministic, lexical — NOT semantic truth)

Since Phase 1, `src/yt_rag.groundedness` classifies each generated answer
against the retrieved context it was grounded on: it extracts claim sentences
and marks each one `supported` (≥ 0.6 of its content words appear in the
context AND every number in the claim appears verbatim in the context) or
`unsupported`; the answer is `empty` (no text), `evasive` (refusal phrase,
e.g. "cannot find it in the transcript"), `supported`,
`partially_supported`, or `unsupported`. The evaluator is fully deterministic
and offline, and is replayed against a committed labeled set
(`experiments/groundedness_set.json`, 9 cases) in every run record.

| Embedder                  | Generation  | Verdicts (12 present items)        | Claim groundedness | Labeled-set agreement | Refusal on absent |
| ------------------------- | ----------- | ---------------------------------- | ------------------ | --------------------- | ----------------- |
| all-MiniLM-L6-v2 (pinned) | StubProvider | 12 supported / 0 partially / 0 unsupported | 1.0 | 9/9 | 0.0 |
| HashEmbedder              | StubProvider | 12 supported / 0 partially / 0 unsupported | 1.0 | 9/9 | 0.0 |

What this DOES measure:

- whether answer text is built from retrieved context (claim-level lexical +
  numeric support), with empty/refusal detection and a fabricated-number
  signal (a number in a claim that is absent from the context);
- that the evaluator itself matches its documented rules on the labeled set
  (9/9 agreement).

What this does NOT measure:

- **Semantic truth.** These are lexical heuristics, NOT a factuality or
  correctness judge: word reuse does not make a claim true (negation, swapped
  entities, and re-ordered claims can all score `supported`), and a true
  paraphrase with different words can score `unsupported`.
- **Contradictions beyond missing numbers/words.** "The limit is 220
  characters" is `supported` by context mentioning 220 even if the truth is
  500; only tokens *absent* from the context are flagged.
- **Question relevance**, whether the answer addresses what was asked.
- **LLM answer quality.** All generation here is `StubProvider`, an
  extractive echo of retrieved chunks, so high supportedness is expected BY
  CONSTRUCTION. These numbers say nothing about LLM answer quality.
- **Refusal behavior.** On the 2 absent-answer questions the refusal rate is
  0.0 because `StubProvider` cannot refuse; its echoes score "supported" even
  though the answer is useless. Measuring refusal requires an LLM provider.

### Qualitative review (open)

**No qualitative LLM review has been run.** `OPENAI_COMPATIBLE_API_KEY` was
not set on the eval machine. `make eval --llm-notes` records qualitative LLM
notes automatically when the key is present. Until then, generation quality
is unjudged by anything semantic, and the Stage 2 qualitative-review box stays
open in `ROADMAP.md`.

### Production limitations

See **Limitations** below for the full list (small hand-labeled eval set, no
tuning yet, flat O(n) retrieval over one transcript, no LLM review, shallow
in-process observability). Nothing here is production-ready: no auth, no TLS,
no multi-user serving, no persisted metrics, no alerting.

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
- **Groundedness numbers are lexical heuristics, not LLM quality.** All
  groundedness numbers above were produced by `StubProvider`, an extractive
  stub that echoes retrieved passages — high supportedness is expected by
  construction. The Phase 1 claim-level evaluator (`src/yt_rag.groundedness`)
  is deterministic and offline, but its labels are NOT semantic truth: no
  contradiction detection beyond numbers/words absent from the context, no
  question-relevance check, and true paraphrases can score unsupported.
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
- **Stage 5/6 observability is local and shallow, and its retrieval-quality
  signal is a PROXY.** Metrics are in-process counters that reset on restart;
  there is no alerting and no persistent metrics store (the Phase 6
  Prometheus TSDB keeps 2 days of these same counters; see "Prometheus +
  Grafana" in Operational notes). The
  "retrieval-quality proxy" (empty-result rate, top score) shows whether
  retrieval returned anything and how confident the vector search was — it is
  not answer quality. Groundedness remains a lexical heuristic (NOT semantic
  truth) and the qualitative LLM review is still open; nothing here relabels
  that.
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
`uvicorn yt_rag.app:app` and `POST /chat` with `{"file": "...", "question": "..."}` (or `"url"` for a live YouTube fetch).

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
correctness. Stage 2's groundedness metric is a deterministic lexical
heuristic (NOT semantic truth) and the qualitative LLM review is open —
Stage 5 does not change or relabel either.

**What could degrade, and the signal that would show it:**

| Degradation                                                                                                        | Signal in logs / metrics                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Transcript format / API changes (youtube-transcript-api breaks, captions removed)                                  | `/chat` 422 with `error_class: "IngestError"` rising in `error_classes` and `error_rate`                                                                 |
| Embedding-model drift (pinned model re-uploaded upstream, local cache vs fresh download differ)                    | `top_score` / `mean_top_score` sliding down with no code change; no hard error                                                                               |
| Index staleness (index built with a different chunk config or embedder than the serving pipeline's current config) | low`top_score` across requests, or a `BundleError`/`ValueError` on load (`error_class` 400s); bundle identity validation is the stronger guard (Stage 3) |
| Empty retrieval                                                                                                    | `empty_result: true` lines; `empty_result_count` / `empty_result_rate` in `/metrics`                                                                     |
| Stub vs real LLM behavior differences                                                                              | stub cannot fail or refuse; a real provider failure surfaces as`error_class: "GenerationError"` with a 500 — watch `error_rate` and `generation_ms`       |

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

### Maintain (Stage 6 — CLI-side; the serving layer is unchanged)

The maintain loop is on-demand and offline, run by hand from the repo root. There is
deliberately **no scheduler, no cron, no CI job**: nothing refreshes or rolls back
automatically, and nothing pages anyone — this is a portfolio maintain drill, not
production MLOps. `yt_rag.app` (and therefore the Docker image and compose) is
**unchanged** from Stage 5: the serving layer still ingests per request and does not read
the maintain pointer, so no image rebuild was needed or done. The pointer and bundles
live under the gitignored `artifacts/` directory.

**Refresh** — re-ingest a transcript, re-embed, and rebuild the FAISS index into a NEW
versioned bundle (previous bundles are never overwritten in place), then move the
`artifacts/CURRENT` pointer at it:

```
python -m yt_rag.maintain --fixture tests/fixtures/teal_chatgpt_linkedin.txt --label v1 \
    --question "how do I optimize my LinkedIn profile with ChatGPT"
# stderr: refreshed: new current bundle .../artifacts/v1-20260907T212022Z
```

`--question` is an optional smoke test: it loads the new bundle through the Stage 3
identity validation and answers a query. Without `--real-embedder` this uses the
deterministic HashEmbedder (fully offline). With `--real-embedder` the rebuild uses the
pinned MiniLM model — that variant is optional AND networked (downloads weights on first
use). `--rollback <name-or-prefix>` and `--list` are the other commands; `--rollback v1`
matches a unique bundle-name prefix and fails loudly on an ambiguous one.

**Rollback** — validate first, then move the pointer:

```
python -m yt_rag.maintain --rollback v1 --question "..."
# stderr: rolled back: current bundle is now .../artifacts/v1-20260907T212022Z
```

The Stage 3 identity validation (`load_bundle`: embedder `model_id` + `dim` against the
manifest) runs **before** the pointer moves. A failed validation (embedder mismatch,
missing files) raises `BundleError` and leaves the pointer untouched — a bundle you
cannot reload can never become current. Tests cover this:
`tests/test_maintain.py::test_rollback_identity_mismatch_leaves_pointer_untouched`.

**Incident runbook (all offline, all executable).** One full write-up with verbatim
outputs: `experiments/incident.md` (bad fixture refresh -> detected via the Stage 5
top-score proxy -> rollback to v1 -> grounded query restored).

| Incident                                                    | Symptom (offline signals)                                                                                                                                                                                        | Runbook action                                                                                                                                                                                                                                             |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Empty results** (Stage 5 empty-result PROXY signal) | `empty_result: true` log line; `empty_result_count`/`empty_result_rate` in `/metrics` — retrieval returned 0 chunks (per-request serving: an empty/unchunkable transcript)                              | Serving builds per request, so there is no stale index to refresh; fix the source transcript, then rebuild a clean bundle if you want one on disk:`python -m yt_rag.maintain --fixture <good.txt> --label v<N>`                                          |
| **Slow retrieval**                                    | `retrieval_ms` / `p95` fields in the `/chat` log line and `/metrics` latency summaries climbing                                                                                                          | Flat index is O(n) per query; at fixture scale this is microseconds. Re-check what changed (corpus size, machine load). Rollback is not a latency tool, but`python -m yt_rag.maintain --list` confirms which bundle is current before comparing builds   |
| **Ingest failure: missing file**                      | `python -m yt_rag.maintain --fixture tests/fixtures/missing_file.txt` -> `error: IngestError: Transcript file not found: ...`, exit code 1, pointer untouched (refresh failed before any bundle was written) | Point`--fixture` at a file that exists (committed fixtures: `tests/fixtures/*.txt`); over HTTP this is the documented 422 with `error_class: "IngestError"`                                                                                          |
| **Ingest/API failure (422)**                          | `/chat` 422 with `error_class: "IngestError"` (no `url`/`file`, bad URL, captions removed, network fetch failure) rising in `/metrics` `error_classes`                                               | Serving maps`IngestError` to 422 (Stage 4 doc). Fix the source or the request body. The maintain CLI is not affected — it reads local fixture files, never YouTube (unless `--url` in `yt_rag.cli`, which is networked)                             |
| **BundleError on embedder mismatch**                  | `error: BundleError: bundle was built with model_id '...', but the expected embedder is '...'` — a rollback or reload refuses the bundle; pointer untouched                                                   | Rebuild with the bundle's embedder:`python -m yt_rag.maintain --fixture <same.txt> --real-embedder --label v<N>` (networked, optional) — or roll back to a bundle that matches the current embedder: `python -m yt_rag.maintain --rollback v<good-N>` |

**Known limitation (stated, not hidden):** the refresh path validates that the new bundle
reloads with the current embedder identity — it cannot detect that the *wrong source
file* was ingested. That failure shows up as collapsing `top_score` on a smoke query (the
Stage 5 proxy), which is exactly how the drill in `experiments/incident.md` is caught and
rolled back.

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

| Variable                      | Read by                                                                                 | When unset                                               | Purpose                                                                                                                                                                                                           |
| ----------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OPENAI_COMPATIBLE_API_KEY` | `OpenAICompatibleProvider` (opt-in real-LLM path) and `make eval` qualitative notes | provider raises`GenerationError`; eval skips LLM notes | API key for the OpenAI-compatible endpoint. The only secret the code can read; passed at run time (`docker compose run --rm -e OPENAI_COMPATIBLE_API_KEY yt-rag ...`), never committed or baked into the image. |
| `YT_RAG_ARTIFACTS_DIR`      | `config.py`                                                                           | `<repo>/artifacts`                                     | Where FAISS indexes and bundles are written.                                                                                                                                                                      |
| `YT_RAG_EVAL_LLM_MODEL`     | `make eval` qualitative notes                                                         | `openai/gpt-4o-mini`                                   | Model used for the optional LLM notes during eval.                                                                                                                                                                |

The real-LLM **base URL and model name are not environment variables**: they
are the `--llm-base-url` / `--llm-model` CLI flags and
`OpenAICompatibleProvider` constructor arguments (default base URL
`https://openrouter.ai/api/v1`). The remaining `config.py` constants (the
embedding model pin, chunk size/overlap, top-k, `EMBEDDING_MODEL_REVISION`)
were audited for Stage 4 and deliberately left in code: they are reproducibility
pins, not secrets or host-specific settings, and `config.py` contains no
hardcoded credentials.

### UI (Phase 4)

A React + Vite + TypeScript workspace lives under `ui/`. It is an addition to
the FastAPI app, not a replacement: the API (`/health`, `/metrics`, `/chat`)
is unchanged and the Python package stays independently usable.

```bash
make ui          # npm install + vite dev server; proxies /health /metrics /chat to localhost:8000 (override with API_PORT)
make ui-build    # production build into ui/dist
make ui-test     # offline vitest + Testing Library suite (mocked fetch, no backend)
```

- Charting library: **recharts** — the recognized React charting option,
  fully client-side, no hosted service, small API surface.
- DEMO mode: the UI ships a bundled pre-rendered sample (clearly labelled
  DEMO) so it is demonstrable with no backend and no network; in that mode it
  never calls `fetch`.
- Provider mode is read live from the `yt_rag_provider_mode` gauge on
  `GET /metrics/prometheus`.
- Groundedness: `/chat` does not return a groundedness field, so the UI shows
  a clearly labelled client-side lexical-overlap heuristic ("Evidence support
  (UI-side heuristic)"). It is NOT the backend's deterministic groundedness
  evaluator.

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

### Prometheus + Grafana (Stage 6 local observability stack)

`docker compose up -d` now starts three services. All host ports are
env-overridable so sibling projects can run in parallel:

| Service | Image | Host port (default) | Override |
| --- | --- | --- | --- |
| yt-rag (FastAPI) | `yt-rag:stage3` (built) | 8000 | `API_PORT` |
| prometheus | `prom/prometheus:v3.9.1` | 9091 | `PROMETHEUS_PORT` |
| grafana | `grafana/grafana-oss:12.3.1` | 3001 | `GRAFANA_PORT` |

- Prometheus scrapes `http://yt-rag:8000/metrics/prometheus` every 5s over the
  compose network (`prometheus.yml` in the repo; the host `API_PORT` mapping
  does not affect scraping). TSDB retention is capped at 2d and storage is
  **intentionally ephemeral**: the `yt_rag_*` metrics are in-process counters
  that reset on app restart, so a persistent TSDB would show fake continuity.
- Grafana auto-provisions (no UI import, no manual clicks):
  - datasource: `provisioning/datasources/prometheus.yml` (Prometheus, uid
    `yt-rag-prom`, proxying `http://prometheus:9090`);
  - dashboard: `dashboards/yt_rag_overview.json` (uid `yt-rag-overview`), loaded
    via `provisioning/dashboards/dashboards.yml`.
- Verified URLs once the stack is up:
  - dashboard: `http://localhost:3001/d/yt-rag-overview`
  - Prometheus targets: `http://localhost:9091/api/v1/targets`
- Grafana login defaults to `admin`/`admin` (local-only stack; the UI asks to
  change it on first interactive login, the HTTP API accepts the defaults).

**Persistence (documented, intentional).** Exactly one named volume,
`grafana-data` (mounted at `/var/lib/grafana`), keeps Grafana UI state across
`compose down`/`up`. Prometheus and app data have **no** volumes, for the
reasons above. `docker compose down` removes containers but keeps
`grafana-data`; `docker compose down -v` would delete it.

**Scope honesty.** The dashboard's retrieval-quality panels (empty-result rate,
mean top score, mean retrieved chunk count) are labeled PROXY in the dashboard
itself: they are not answer quality. With the default offline path, retrieval
returns up to `top_k` chunks whenever the index is non-empty, so the
empty-result series legitimately has no data in a healthy run; the panel shows
its "No data" state rather than a misleading zero-rate line.
