# AGENTS.md - youtube_app

Project instructions for AI coding agents (Prime Agent, Copilot, Claude Code, etc.) working in this repository.

## Purpose

Turn `youtube_app` into a portfolio-grade, full-lifecycle RAG application. The goal is to demonstrate the complete AI engineering loop: build, ship, deploy, monitor, and maintain a retrieval-augmented generation (RAG) system over YouTube transcripts.

This is a personal portfolio project, not production work. It must be honest, rigorous, and reproducible. Do not oversell capabilities that are not implemented.

## Current state (as of Sep 2026)

- Dormant since Sep 2024.
- Existing functionality: transcript ingestion from a YouTube URL, summary generation, and chat over the transcript.
- Known gap: the app is described as RAG but has no real retrieval layer. This must be fixed with genuine chunking, embeddings, vector search, and top-k retrieval.

## Non-negotiable requirements

1. **Real RAG, not a wrapper.** Implement actual retrieval: chunking, embedding, vector store, top-k retrieval, and a generation step that uses retrieved context. A direct LLM call with no retrieval is not acceptable.
2. **Evaluation before optimization.** Add retrieval-quality metrics (hit rate, mean reciprocal rank) and generation-quality metrics before tuning anything.
3. **Reproducibility.** Pin dependencies. Provide a lockfile and a documented setup path.
4. **Honest documentation.** Use the structure: Motivation, Method, Results, Limitations, Operational notes. Never claim a capability that is not implemented.
5. **Tests.** A test suite must exist and pass. No untested claims.

## Tech stack guidance

The user is agnostic to stack. Prefer boring, well-supported, widely understood tools that a hiring manager will recognize. Python is the default for the ML/RAG core. Suggested defaults (change only with justification):

- Language: Python 3.11+
- Embeddings + vector store: sentence-transformers + FAISS (local, free, reproducible) OR a managed option if a public deployment is chosen
- Generation: an LLM API (OpenAI-compatible) with a documented provider abstraction
- Serving: FastAPI
- Packaging: Docker + docker-compose for local reproducible deployment
- CI/CD: GitHub Actions
- Monitoring: structured logging + a lightweight metrics endpoint; Prometheus/Grafana optional

## Working conventions

- Keep the model-facing tool surface small. Prefer a persistent Python REPL as the control environment.
- Run project commands from the repo root.
- After any code change, run the test suite and the linter.
- Update ROADMAP.md as work progresses. Mark completed items.
- Use git branches for each lifecycle stage. Commit with clear messages.
- Do not commit secrets, API keys, or large model artifacts. Use .gitignore and environment variables.

## Lifecycle stage ownership

This repo owns the **RAG / GenAI serving + monitoring** pillar of the portfolio. Do not duplicate the forecasting or optimization work that belongs to the sibling repos.

## Context files

- `ROADMAP.md` in this directory is the source of truth for planned and completed work.
- Global instructions may also be loaded from `~/.prime/agent/AGENTS.md`; project instructions here take precedence for this repo.