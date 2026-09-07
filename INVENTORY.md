# INVENTORY.md — youtube_app legacy review

Stage 0 inventory of `legacy/` (read-only reference, dormant since Sep 2024) mapped
against `AGENTS.md` requirements and `ROADMAP.md` items. No legacy code was modified.

## legacy/ contents

| Path | What it is | Maps to | Notes / gaps |
| --- | --- | --- | --- |
| `README.md` | Docs for the old Flask + React "YouTube Transcript Analyzer" | Stage 0 README skeleton (structure differs) | Claims RAG-style Q&A, but answers are single LLM calls over the **full** transcript. No retrieval layer. |
| `requirements.txt` | Fully pinned deps (Flask 3.0.3, openai 1.43.0, anthropic 0.34.1, google-generativeai 0.7.2, torch 2.4.0, transformers 4.44.2, youtube-transcript-api 0.6.2) | Stage 0 lockfile requirement | Pinned, but stale (Aug 2024) and heavy: torch/transformers pulled in only for two optional "llama" paths (BART summarization, RoBERTa QA). No FAISS, no sentence-transformers, no vector store. |
| `backend/main.py` | Monolith: transcript fetch (youtube-transcript-api), URL/video-ID extraction, summarize + answer via GPT/Claude/Gemini/local BART, Flask routes (`/api/transcript`, `/api/summarize`, `/api/answer`, `/test`, `/test-form`) | Stage 1 ingestion, Stage 4 serving | Real gap AGENTS.md calls out: **no chunking, no embeddings, no vector store, no top-k retrieval**. `answer_with_*` stuffs the entire transcript into one prompt. Crashes at import if API keys are missing. Debug `print()` logging. |
| `backend/app.py` | 5-line runner that imports `app` from `main` (run with `debug=True`) | Stage 4 FastAPI serving | Will be replaced; Flask is not the documented Stage 4 target. |
| `backend/test_main.py` | 3 unittest cases: video-ID regex, summarize, answer | Stage 0 test harness | Summarize/answer tests hit real LLM APIs (need keys + network); unusable in CI. Module import fails without keys. Nothing tests retrieval (there is none). |
| `backend/transcripts/` (2 .txt) | Cached fetched transcripts | Stage 1 ingestion | Small sample data; fine as eval fixture candidates (Stage 2). |
| `transcripts/` (2 .txt) | More cached transcripts at repo root | Same | Duplicate/unstructured cache location. |
| `frontend/` | Create React App app (React, Material-UI, Axios): URL input, model picker, summarize/ask forms | Not in scope for Stage 0 | Old CRA stack; decision on a new UI (or API-only) deferred to Stage 4. |
| `package.json`, `package-lock.json` (root) | CRA tooling only (`@babel/plugin-proposal-private-property-in-object` dev dep) | — | Root-level Node noise; not needed for the Python pillar. |
| `.gitignore` | Broad Python/Node/env/IDE ignores | Stage 0 .gitignore | Good reference; new root `.gitignore` mirrors the Python-relevant subset. |
| `.git/` | Shallow git history of the original GitHub clone | — | Reference only; the new repo starts fresh. |

## Gaps AGENTS.md explicitly calls out

1. **"RAG" in name only.** No chunking, embeddings, vector store, or top-k retrieval
   anywhere in `legacy/`. Summarize/answer are direct LLM calls with the whole
   transcript in the prompt. This is the core Stage 1 build item.
2. **No evaluation.** No retrieval metrics (hit rate, MRR) or generation-quality
   checks. Stage 2 requires eval before any tuning.
3. **No reproducible setup path.** `requirements.txt` is pinned but there is no
   lockfile workflow, no Makefile, no CI, no Docker. Stage 0/3 acceptance requires
   `git clone && make setup && make test`.
4. **No honest test suite.** Existing tests call paid APIs and fail without keys.
5. **Secrets handling is fragile.** Module-level hard failure when keys are missing;
   `.env` handled via dotenv, but no env-based config pattern worth keeping.

## What is worth keeping (reference only)

- Video-ID extraction regex patterns (`extract_video_id`) — reusable logic for
  Stage 1 ingestion.
- `youtube-transcript-api` as the ingestion library choice.
- Cached transcript `.txt` files as small, offline fixture/eval candidates.

## Stack decision (Stage 0)

Python 3.11+ package `yt_rag` under `src/`, per parent AGENTS.md stack defaults.
Embeddings/vector store and generation provider choices are Stage 1 decisions and
are intentionally **not** implemented in Stage 0.
