# Local packaging for yt_rag (Stage 3). Installs the single pinned lockfile
# (requirements-lock.txt) plus the package itself. No API keys or secrets are
# baked in; configuration stays environment-based.
# Python 3.12 (not 3.11): the universal lock resolved numpy 2.5.3, which
# requires Python >= 3.12 (see README Operational notes).
# Note: requirements-lock.txt is universal and includes torch on Linux, so the
# image is large (several GB). See README "Operational notes" for the tradeoff.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first for layer caching; single lockfile, no second lock.
COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

# The package itself (deps already satisfied by the lock).
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps .

# Committed transcript fixtures so the offline smoke path needs no network:
# the default app uses HashEmbedder + StubProvider (no model downloads, no keys).
COPY tests/fixtures/ /app/tests/fixtures/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

CMD ["uvicorn", "yt_rag.app:app", "--host", "0.0.0.0", "--port", "8000"]
