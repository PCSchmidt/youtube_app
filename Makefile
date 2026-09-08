# Portable across Linux and Windows (Git Bash).
ifeq ($(OS),Windows_NT)
    VENV_PY := .venv/Scripts/python.exe
else
    VENV_PY := .venv/bin/python
endif

.PHONY: setup test lint format eval clean ui ui-build

setup:
	python -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements-lock.txt
	$(VENV_PY) -m pip install -e . --no-deps

test:
	$(VENV_PY) -m pytest -q
	$(VENV_PY) -m ruff check .

lint:
	$(VENV_PY) -m ruff check .
	$(VENV_PY) -m ruff format --check .

format:
	$(VENV_PY) -m ruff format .

# Stage 2 evaluation. NOT part of `make test`: `make eval` may download the
# pinned MiniLM weights on first use (network) and, if OPENAI_COMPATIBLE_API_KEY
# is set, also records qualitative LLM notes. Run records land in experiments/runs/.
eval:
	$(VENV_PY) -m yt_rag.eval --embedder both --llm-notes

# Phase 4 UI (React + Vite + TypeScript under ui/). Dev server proxies
# /health /metrics /chat to the FastAPI backend; API_PORT overrides 8000.
ui:
	npm --prefix ui install --no-audit --no-fund
	npm --prefix ui run dev

ui-build:
	npm --prefix ui install --no-audit --no-fund
	npm --prefix ui run build

ui-test:
	npm --prefix ui run test

clean:
	rm -rf .venv .pytest_cache .ruff_cache dist build *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
