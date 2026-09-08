## UI (Phase 4, minimal)

A React + Vite + TypeScript workspace lives under `ui/`. It is an addition to
the FastAPI app, not a replacement: the API (`/health`, `/metrics`,
`/chat`) is unchanged.

```bash
make ui            # npm install + vite dev server (proxies /health /metrics /chat to localhost:8000, override with API_PORT)
make ui-build      # production build into ui/dist
# or, inside ui/: npm run dev | npm run build | npm test
```

- Charting: **recharts** (chosen because it is the recognized React charting
  option, purely client-side, no hosted service, small API surface).
- DEMO mode: the UI ships a bundled pre-rendered sample (clearly labelled
  DEMO) so it is demonstrable with no backend and no network. It never
  calls `fetch` in that mode.
- Provider mode is read live from the `yt_rag_provider_mode` gauge on
  `GET /metrics/prometheus`.
- Groundedness: `/chat` does not return a groundedness field, so the UI shows
  a clearly labelled client-side lexical-overlap heuristic ("Evidence support
  (UI-side heuristic)"). It is NOT the backend groundedness evaluator.
- Tests: `npm test` runs vitest + Testing Library offline (mocked fetch, no
  network, no backend).
