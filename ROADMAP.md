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
- [ ] Generation metrics: faithfulness/groundedness check, plus a qualitative review.
- [x] Record baseline numbers in an `experiments/` run log with date, config, and results.

**Acceptance:** Baseline metrics are recorded and reproducible. No tuning happens before this.

## Stage 3 - Ship (versioned, reproducible)

- [ ] Model + code versioning: tag releases; pin the embedding model version.
- [ ] Model registry / artifact bundle: a documented way to store and load the index + model.
- [ ] `requirements.lock` and a reproducible build path.
- [ ] Containerize the app (Dockerfile) and provide `docker-compose.yml` for local run.
- [ ] CI/CD pipeline that builds, tests, and produces a tagged artifact.

**Acceptance:** A tagged release can be rebuilt and run reproducibly from the artifact bundle.

## Stage 4 - Deploy

- [ ] FastAPI serving layer with health check and a documented API.
- [ ] Deploy target decision: local Docker Compose (minimum) or a public endpoint (optional, if cost is acceptable).
- [ ] Environment-based configuration (no hardcoded secrets).
- [ ] Document the deployment runbook.

**Acceptance:** The app runs from the container and responds to health + query endpoints.

## Stage 5 - Monitor

- [ ] Structured logging of requests, retrieval latency, and generation latency.
- [ ] Metrics endpoint exposing: request count, latency percentiles, error rate, retrieval quality proxy (e.g., empty-result rate).
- [ ] Optional: Prometheus/Grafana dashboard.
- [ ] Document "what could degrade" (e.g., transcript format changes, embedding model drift, index staleness).

**Acceptance:** A reviewer can see how the app is observed and what signals would indicate a problem.

## Stage 6 - Maintain

- [ ] Refresh/retraining path: how to re-ingest, re-embed, and rebuild the index.
- [ ] Rollback path: how to revert to a previous index/model version.
- [ ] Runbook for common incidents (empty results, slow retrieval, API failures).
- [ ] One documented incident write-up (real or realistic) showing the maintain loop.

**Acceptance:** The maintain loop is documented and executable, not just described.

## Portfolio presentation

- [ ] README tells the full lifecycle story with real numbers.
- [ ] Link the repo from `pcschmidt.github.io`.
- [ ] Prepare a 3-sentence interview arc per lifecycle stage.

## Definition of done

All stages complete, tests green, evaluation numbers recorded, deployment reproducible, monitoring documented, and the README honestly reflects what is implemented.