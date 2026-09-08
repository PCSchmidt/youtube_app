# ROADMAP.md - youtube_app

Full-lifecycle RAG application over YouTube transcripts. This roadmap drives the build, ship, deploy, monitor, and maintain stages. Each item has concrete acceptance criteria so work is verifiable, not aspirational.

## North star

A reviewer can clone this repo, run one command, and see a working RAG app with real retrieval, real evaluation numbers, a reproducible deployment, and a documented monitoring story. The README tells the full lifecycle story honestly.

## Stage 0 - Foundation (prereq)

- [x] Inspect existing code and inventory what works vs. what is broken.
- [x] Decide stack (see AGENTS.md defaults) and document the choice.
- [x] Set up Python 3.11+ environment with pinned dependencies and a lockfile.
- [x] Add `.gitignore` for secrets, artifacts, and model caches.
- [x] Establish a test harness and CI (GitHub Actions) that runs tests + lint.
- [x] Write the README skeleton with the Motivation / Method / Results / Limitations / Operational notes structure.

**Acceptance:** `git clone && make setup && make test` succeeds on a clean machine.

## Stage 1 - Build (real RAG)

- [x] Transcript ingestion: robust YouTube transcript fetch with error handling.
- [x] Chunking strategy: document and implement (e.g., recursive character splitting with overlap). Justify the choice.
- [x] Embeddings: sentence-transformers model, pinned version.
- [x] Vector store: FAISS index with save/load.
- [x] Retrieval: top-k search with a documented similarity metric.
- [x] Generation: LLM call that uses retrieved context; provider abstraction so it is swappable.
- [x] Chat endpoint that ties ingestion -> retrieval -> generation together.

**Acceptance:** A query returns an answer grounded in retrieved transcript chunks, and the retrieval path is exercised end to end.

## Stage 2 - Evaluate (before optimizing)

- [x] Build a small labeled eval set (queries + expected relevant chunks/answers).
- [x] Retrieval metrics: hit rate, mean reciprocal rank (MRR).
- [x] Generation metrics: deterministic groundedness check — claim-level
  supported/partially-supported/unsupported verdicts plus empty/evasive
  detection (`src/yt_rag.groundedness`), replayed against a committed labeled
  set (`experiments/groundedness_set.json`). LEXICAL HEURISTICS, NOT semantic
  truth; see the limitations in `experiments/baseline_log.md` and the module
  docstring.
- [ ] Qualitative review: a qualitative LLM review of generated answers is
  still open. It requires `OPENAI_COMPATIBLE_API_KEY` at eval time; the gap is
  recorded, not faked (`yt_rag.eval.maybe_llm_notes`).
- [x] Record baseline numbers in an `experiments/` run log with date, config, and results.

**Acceptance:** Baseline metrics are recorded and reproducible. No tuning happens before this.

## Stage 3 - Ship (versioned, reproducible)

- [x] Model + code versioning: tag releases; pin the embedding model version.
- [x] Model registry / artifact bundle: a documented way to store and load the index + model.
- [x] `requirements.lock` and a reproducible build path.
- [x] Containerize the app (Dockerfile) and provide `docker-compose.yml` for local run.
- [x] CI/CD pipeline that builds, tests, and produces a tagged artifact.

Stage 3 note (scope and honesty): the tag (`stage3-v0.1.0`) is local
bookkeeping, not a pushed release. CI builds the Docker image as build proof
and pushes to no registry. Compose is local-only — this is **not** a production
deploy. The artifact bundle stores identity metadata, not weights; the index
itself stays gitignored. The lock resolved `numpy 2.5.3`, which requires
Python >= 3.12, so the Docker image and CI now use Python 3.12 (documented in
README). Stage 3 acceptance holds: a bundle is saved, reloaded, and queried
with identity validation (tests/test_bundle.py), and the image was built and
smoke-tested locally (`docker compose up` + `/health` + fixture-backed
`/chat`, all offline).

**Acceptance:** A tagged release can be rebuilt and run reproducibly from the artifact bundle.

## Stage 4 - Deploy

- [x] FastAPI serving layer with health check and a documented API.
- [x] Deploy target decision: local Docker Compose (minimum) or a public endpoint (optional, if cost is acceptable).
- [x] Environment-based configuration (no hardcoded secrets).
- [x] Document the deployment runbook.

Stage 4 note (scope and honesty): this stage is docs-only on top of existing
code — the serving layer itself is unchanged since Stage 1 (`/health` +
`/chat`, stub or real provider). Local Docker Compose is the accepted deploy
target; a public endpoint was declined (cost, portfolio-not-prod), so there is
no TLS, auth, multi-user serving, or registry push. The runbook in README
Operational notes was verified end to end on this branch: `docker compose up`
(reusing the existing `yt-rag:stage3` image — no torch-heavier rebuild),
`/health` 200, fixture-backed `/chat` 200, `docker compose down`. The env-var
audit found three optional variables and no hardcoded secrets.

**Acceptance:** The app runs from the container and responds to health + query endpoints.

## Stage 5 - Monitor

- [x] Structured logging of requests, retrieval latency, and generation latency.
- [x] Metrics endpoint exposing: request count, latency percentiles, error rate, retrieval quality proxy (e.g., empty-result rate).
- [x] Prometheus-compatible exposition endpoint: `GET /metrics/prometheus`
  (text exposition v0.0.4, stdlib-only writer, no new dependency). Generic
  families only — `yt_rag_requests_total`, `yt_rag_errors_total`,
  `yt_rag_request_latency_seconds` histogram, `yt_rag_up` — with bounded
  route-template/status/error-class labels; the JSON `GET /metrics` contract
  is unchanged. In-process and reset on restart, like the JSON counters.
- [ ] Optional: Prometheus/Grafana dashboard (still unchecked: no scraper,
  no storage, no dashboards are stood up — only the exposition text).
- [x] Document "what could degrade" (e.g., transcript format changes, embedding model drift, index staleness).

Stage 5 note (scope and honesty): observability is local and in-process — one
JSON log line per request (stdlib logging, stdout), a `GET /metrics`
endpoint of plain process counters that reset on restart, and (Phase 2) a
`GET /metrics/prometheus` text-exposition endpoint written by hand with the
stdlib only — still in-process, still no scraper or storage. Phase 3 adds the
app-specific families there (retrieval/generation latency histograms,
empty-result counter, mean_top_score / mean_retrieved_count PROXY gauges,
question-length histogram, provider-mode gauge) with endpoint-only bounded
labels; they mirror the JSON snapshot semantics but are NOT answer quality. The retrieval-quality
signal is a PROXY (empty-result rate, top score), not a quality measure:
Stage 2 groundedness is a lexical heuristic, NOT semantic truth, and the
qualitative LLM review is still open. No new dependencies, no extra containers, no dashboards, no alerting;
the optional Prometheus/Grafana box is left unchecked because nothing was
stood up.

**Acceptance:** A reviewer can see how the app is observed and what signals would indicate a problem.

## Stage 6 - Maintain

- [x] Refresh/retraining path: how to re-ingest, re-embed, and rebuild the index.
- [x] Rollback path: how to revert to a previous index/model version.
- [x] Runbook for common incidents (empty results, slow retrieval, API failures).
- [x] One documented incident write-up (real or realistic) showing the maintain loop.

Stage 6 note (scope and honesty): this is an **index rebuild** loop, not retraining —
there is no model to retrain, and it is not gated on the sibling stock app's PSI/KS drift
checks. Refresh = re-ingest a committed fixture + re-embed + rebuild the FAISS index into
a NEW versioned bundle (`python -m yt_rag.maintain --fixture <txt> --label vN`), rollback
= move the `artifacts/CURRENT` pointer back only after Stage 3 identity validation passes.
It runs on demand from the CLI; there is deliberately no scheduler, no cron, no registry,
no alerting. The serving layer and compose are unchanged (the app ingests per request and
does not read the pointer), so no image rebuild was needed. The offline default uses
HashEmbedder + StubProvider + committed fixtures; the real-embedder rebuild is the same
command with `--real-embedder` and is optional AND networked. Runbook: README Operational
notes; executed incident with verbatim outputs: `experiments/incident.md`
(top_score 0.257 -> 0.041 on a wrong-fixture refresh, rollback to v1, grounded query
restored, and a failed identity validation shown to leave the pointer untouched).
Stage 2's groundedness remains a lexical heuristic (NOT semantic truth) and the qualitative LLM review stays open — Stage 6
does not touch either.

**Acceptance:** The maintain loop is documented and executable, not just described.

## Portfolio presentation

- [ ] README tells the full lifecycle story with real numbers.
- [ ] Link the repo from `pcschmidt.github.io`.
- [ ] Prepare a 3-sentence interview arc per lifecycle stage.

## Definition of done

All stages complete, tests green, evaluation numbers recorded, deployment reproducible, monitoring documented, and the README honestly reflects what is implemented.