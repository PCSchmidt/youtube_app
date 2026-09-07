"""Central configuration for the yt_rag pipeline.

The embedding model is pinned by name here so every build of the index is
reproducible. The exact sentence-transformers *library* version is pinned in
requirements-lock.txt; if the model is ever re-uploaded upstream, record the
HF revision hash in EMBEDDING_MODEL_REVISION to stay bit-reproducible.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Embedding model (pinned) ------------------------------------------------
# all-MiniLM-L6-v2: small (22M params), fast, MIT-licensed, 384-dim vectors,
# and the most widely recognized sentence-transformers model in production use.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_DIM = 384
# Optional HF revision pin; None means "resolve to the default branch tip at
# first download" (which is then cached locally). Set a commit hash to pin.
EMBEDDING_MODEL_REVISION: str | None = None

# --- Chunking ----------------------------------------------------------------
# 800 chars ~ 150-200 tokens: several complete sentences, well inside the
# model's 256-token window. 150-char overlap keeps boundary context continuous.
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 150

# --- Retrieval / generation --------------------------------------------------
DEFAULT_TOP_K = 4

# --- Artifacts ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.environ.get("YT_RAG_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
