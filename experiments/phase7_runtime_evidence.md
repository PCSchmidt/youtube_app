# Phase 7 runtime + visual validation evidence (yt_rag observability stack)

Date: 2026-09-08. Host: Windows Git Bash, Docker 29.7.2, Compose v5.5.0.
Stack: `docker compose up -d` -> yt-rag (8000), prometheus (9091), grafana (3001).
Ports env-overridable: `API_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT`.

## Traffic generation (offline fixtures, no network, no keys)

```
# 53 successful POST /chat using committed fixtures (container paths):
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"file": "/app/tests/fixtures/teal_chatgpt_linkedin.txt",
       "question": "how do I optimize my LinkedIn profile with ChatGPT"}'
# 5 classified 422 IngestError requests: {"question": "no source given"}
# repeated GET /health
```

## Checks (all pass at time of writing)

1. API health: `GET :8000/health` -> 200 `{"status": "ok"}`; container
   healthy (Docker healthcheck).
2. Prometheus target: `GET http://localhost:9091/api/v1/targets` ->
   `http://yt-rag:8000/metrics/prometheus` health=`up`, lastError empty.
3. Instant query: `GET :9091/api/v1/query?query=yt_rag_requests_total` ->
   real values, e.g. `{endpoint="/chat",status="200"} = 53`,
   `{endpoint="/chat",status="422"} = 5`, `{endpoint="/health"} = 29`.
   Domain queries also return data: chat rate ~0.047 req/s (5m),
   retrieval p95 0.00475s, `yt_rag_mean_top_score` 0.207,
   `yt_rag_mean_retrieved_count` 4, error classes `IngestError` increasing.
4. Grafana datasource health: `GET admin:admin@localhost:3001/api/datasources/
   uid/yt-rag-prom/health` -> 200, `{"status":"OK"}` (defaults admin/admin;
   noted in README).
5. Dashboard UID: `GET :3001/api/dashboards/uid/yt-rag-overview` -> 200,
   title "yt_rag - RAG serving overview", 13 panels.
6. Query through Grafana itself (`POST /api/ds/query`, datasource
   `yt-rag-prom`): returns real values (e.g. total request rate 0.29 req/s).

## Test / lint / UI

- `make test`: 134 passed in 2.10s; ruff check: All checks passed.
- `make lint`: ruff check + `ruff format --check`: 40 files already formatted.
- `npm --prefix ui run build` (tsc --noEmit && vite build): OK, built in 2.87s.

## Code-level visual checks

- Loading/error states: UI has explicit `idle|loading|success|error` status
  machine + `StatusBanner`; dashboard panels set `noValue: "No data"` (errors
  table: "No errors in range") instead of misleading zeros.
- Units: req/s (reqps), percent (error/empty rates), seconds (latency),
  percentunit (mean top score 0-1), short (chunk counts, error class counts).
- Horizontal-scroll risk: dashboard is a fixed 24-col grid (Grafana handles
  column stacking on narrow widths); recent-errors table is the widest panel
  but uses 4 short columns. UI: `max-width: 1100px` workspace + 800px
  breakpoint; low risk, not screenshot-verified here (parent captures).

## Known no-data (honest)

- `yt_rag_empty_results_total` has no series in a healthy run: the offline
  retriever always returns up to top_k chunks when the index is non-empty.
  The empty-result panel shows its "No data" state; behavior documented.
- After a burst-then-idle traffic pattern, `rate()` windows decay to 0 until
  new requests arrive; latency percentile queries return values only while
  the underlying counters increase within the window.

## State

Stack left RUNNING for central screenshots (do `docker compose down` after;
named volume `grafana-data` persists, documented in README).
