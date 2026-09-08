"""Versioned artifact bundle (Stage 3): FAISS index + chunk metadata + manifest.

A bundle is a directory containing:

- ``index.faiss`` and ``chunks.json`` (the FAISS store, exactly as
  :meth:`yt_rag.vectorstore.FaissVectorStore.save` writes them), and
- ``manifest.json``: identity metadata only — embedding model id + dim,
  chunking config, top-k, package version, creation time, and git commit.
  No model weights are stored: the real model stays in its local cache
  (gitignored) and is identified, not shipped, by the manifest.

Two reproduction paths, both documented in the README:

- **Rebuild**: re-run ingest + embed with the pinned config from a fresh clone
  (the manifest tells you exactly what produced the index).
- **Reload**: load the bundle directory and query it with an embedder whose
  model_id and dim match the manifest.

The manifest is small enough to commit an example of; the index itself stays
under the gitignored ``artifacts/`` directory.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yt_rag
from yt_rag.config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_DIM,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
)
from yt_rag.vectorstore import FaissVectorStore

if TYPE_CHECKING:
    from yt_rag.embeddings import EmbeddingProvider

MANIFEST_NAME = "manifest.json"
BUNDLE_FORMAT_VERSION = 1


class BundleError(Exception):
    """Raised when a bundle is missing files or its identity does not match."""


def embedder_model_id(embedder: EmbeddingProvider) -> str:
    """Identity string for the embedder that built (or will query) an index."""
    explicit = getattr(embedder, "model_id", None)
    return str(explicit) if explicit else type(embedder).__name__


def git_commit() -> str | None:
    """Short commit hash of the working tree, or None outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_manifest(
    model_id: str,
    dim: int,
    top_k: int = DEFAULT_TOP_K,
    revision: str | None = EMBEDDING_MODEL_REVISION,
) -> dict:
    """Identity manifest for an index. Metadata only — never weights."""
    return {
        "bundle_format": BUNDLE_FORMAT_VERSION,
        "package_version": yt_rag.__version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "embedder": {"model_id": model_id, "dim": dim, "revision": revision},
        "chunking": {"size_chars": CHUNK_SIZE_CHARS, "overlap_chars": CHUNK_OVERLAP_CHARS},
        "retrieval": {
            "top_k": top_k,
            "metric": "cosine (IndexFlatIP over L2-normalized vectors)",
        },
    }


def save_bundle(
    store: FaissVectorStore,
    directory: str | Path,
    *,
    model_id: str,
    top_k: int = DEFAULT_TOP_K,
) -> Path:
    """Write the FAISS store plus ``manifest.json`` into ``directory``."""
    directory = Path(directory)
    store.save(directory)  # index.faiss + chunks.json
    manifest = build_manifest(model_id=model_id, dim=store.dim, top_k=top_k)
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return directory


def load_bundle(
    directory: str | Path,
    *,
    expected_model_id: str | None = EMBEDDING_MODEL_NAME,
    expected_dim: int | None = EMBEDDING_MODEL_DIM,
) -> tuple[FaissVectorStore, dict]:
    """Load a bundle and validate its identity against the expected embedder.

    Pass ``expected_model_id=None`` (and/or ``expected_dim=None``) to skip that
    check. Returns the loaded store and the manifest dict.
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise BundleError(f"no {MANIFEST_NAME} in {directory}; not an artifact bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    embedder_meta = manifest.get("embedder", {})
    if expected_model_id is not None and embedder_meta.get("model_id") != expected_model_id:
        raise BundleError(
            f"bundle was built with model_id {embedder_meta.get('model_id')!r}, "
            f"but the expected embedder is {expected_model_id!r}; rebuild the index "
            "with the bundle's model before querying it"
        )
    if expected_dim is not None and embedder_meta.get("dim") != expected_dim:
        raise BundleError(f"bundle dim {embedder_meta.get('dim')} != expected dim {expected_dim}")

    store = FaissVectorStore.load(directory)  # raises IndexStateError if index missing
    if store.dim != embedder_meta.get("dim"):
        raise BundleError(
            f"index dim {store.dim} does not match manifest dim {embedder_meta.get('dim')}"
        )
    return store, manifest
