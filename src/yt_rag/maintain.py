"""Index maintenance (Stage 6): offline refresh + pointer rollback.

This module is deliberately CLI-side and offline. It does NOT retrain anything:
"refresh" means re-ingest a transcript fixture, re-embed it, and rebuild the
FAISS index into a NEW versioned artifact bundle (never overwriting a previous
bundle in place), then move a ``CURRENT`` pointer file at it. "Rollback" moves
the pointer back to a previous bundle — but only after the Stage 3 identity
validation (:func:`yt_rag.bundle.load_bundle`) succeeds; a failed validation
leaves the pointer untouched, so a bad bundle can never become current.

The pointer is a one-line file ``<artifacts>/CURRENT`` containing the bundle
directory name relative to the artifacts root. Bundles stay gitignored under
``artifacts/``.

Commands (offline; HashEmbedder + StubProvider + committed fixtures):

    python -m yt_rag.maintain --fixture tests/fixtures/teal_chatgpt_linkedin.txt --label v1
    python -m yt_rag.maintain --fixture tests/fixtures/game_dev_ai_paper.txt --label v2
    python -m yt_rag.maintain --list
    python -m yt_rag.maintain --rollback v1 --question "..."

``--real-embedder`` rebuilds with the pinned MiniLM model instead of the hash
embedder; that variant downloads model weights on first use (network) and is
optional. The real-embedder rebuild is the same command plus that flag —
everything here is documented and runnable offline by default.

This is on-demand maintenance run by hand, not production MLOps: no scheduler,
no registry, no alerting. The serving layer (``yt_rag.app``) is unchanged — it
still ingests per request, so the maintain loop is exercised from the CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from yt_rag.bundle import BundleError, embedder_model_id, load_bundle
from yt_rag.config import ARTIFACTS_DIR
from yt_rag.embeddings import HashEmbedder
from yt_rag.errors import IngestError
from yt_rag.pipeline import RAGPipeline

POINTER_NAME = "CURRENT"


def _artifacts_dir(artifacts_dir: str | Path | None) -> Path:
    return Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR


def pointer_path(artifacts_dir: str | Path | None = None) -> Path:
    """Path of the CURRENT pointer file for an artifacts root."""
    return _artifacts_dir(artifacts_dir) / POINTER_NAME


def read_pointer(artifacts_dir: str | Path | None = None) -> Path | None:
    """Resolve the current bundle directory, or None if the pointer is unset.

    Raises BundleError if the pointer names a directory that does not exist.
    """
    p = pointer_path(artifacts_dir)
    if not p.is_file():
        return None
    name = p.read_text(encoding="utf-8").strip()
    if not name:
        raise BundleError(f"{p} is empty; rewrite it with a valid bundle name or refresh")
    target = _artifacts_dir(artifacts_dir) / name
    if not target.is_dir():
        raise BundleError(f"current pointer {p} names missing bundle directory {name!r}")
    return target


def _versioned_dir(artifacts_dir: Path, label: str) -> Path:
    """A fresh, non-existent versioned bundle directory. Never reuses a name."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = artifacts_dir / f"{label}-{stamp}"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = artifacts_dir / f"{label}-{stamp}-{suffix}"
    return candidate


def refresh(
    fixture: str | Path,
    *,
    artifacts_dir: str | Path | None = None,
    label: str = "bundle",
    embedder=None,
    provider=None,
    top_k: int | None = None,
) -> Path:
    """Re-ingest + re-embed a transcript and rebuild the index as a NEW bundle.

    The bundle is written to a fresh versioned directory under the artifacts
    root (previous bundles are never overwritten), validated with the Stage 3
    ``load_bundle`` identity check, and only then made current via the pointer.

    Returns the new current bundle directory.
    """
    artifacts = _artifacts_dir(artifacts_dir)
    embedder = embedder or HashEmbedder()
    kwargs = {"embedder": embedder}
    if provider is not None:
        kwargs["provider"] = provider
    if top_k is not None:
        kwargs["top_k"] = top_k
    pipeline = RAGPipeline(**kwargs)
    pipeline.ingest_from_file(fixture)  # IngestError on missing/empty file

    out = _versioned_dir(artifacts, label)
    pipeline.save_bundle(out)

    # Validate before the pointer moves: a bundle we cannot reload with the
    # same embedder identity must never become current.
    load_bundle(out, expected_model_id=embedder_model_id(embedder), expected_dim=embedder.dim)

    artifacts.mkdir(parents=True, exist_ok=True)
    pointer_path(artifacts).write_text(out.name + "\n", encoding="utf-8")
    return out


def rollback(
    target: str | Path,
    *,
    artifacts_dir: str | Path | None = None,
    embedder=None,
) -> Path:
    """Point CURRENT back at a previous bundle after validating its identity.

    ``target`` is a bundle directory name relative to the artifacts root (as
    printed by ``--list``) or a path to a bundle directory. The Stage 3
    identity validation runs FIRST; if it fails (embedder model_id/dim
    mismatch, missing files), the pointer is left untouched.
    """
    artifacts = _artifacts_dir(artifacts_dir)
    embedder = embedder or HashEmbedder()
    candidate = Path(target)
    if not candidate.is_dir():
        candidate = artifacts / str(target)
    if not candidate.is_dir():
        # Accept a unique name prefix (e.g. "v1" for "v1-20260908T120000Z").
        matches = [d for d in list_bundles(artifacts) if d.name.startswith(str(target))]
        if len(matches) == 1:
            candidate = matches[0]
        elif len(matches) > 1:
            raise BundleError(
                f"ambiguous rollback target {target!r}: matches "
                + ", ".join(m.name for m in matches)
                + "; give the full bundle name"
            )
        else:
            raise BundleError(f"no such bundle directory: {target}")

    load_bundle(candidate, expected_model_id=embedder_model_id(embedder), expected_dim=embedder.dim)

    pointer = pointer_path(artifacts)
    pointer.write_text(candidate.name + "\n", encoding="utf-8")
    return candidate


def list_bundles(artifacts_dir: str | Path | None = None) -> list[Path]:
    """All bundle directories (containing manifest.json), sorted by name."""
    artifacts = _artifacts_dir(artifacts_dir)
    if not artifacts.is_dir():
        return []
    return sorted(d for d in artifacts.iterdir() if d.is_dir() and (d / "manifest.json").is_file())


def ask_current(question: str, *, artifacts_dir: str | Path | None = None, embedder=None) -> dict:
    """Load the current bundle (with identity validation) and answer a query."""
    embedder = embedder or HashEmbedder()
    current = read_pointer(artifacts_dir)
    if current is None:
        raise BundleError(
            "no CURRENT pointer; run a refresh first (python -m yt_rag.maintain --fixture ...)"
        )
    pipeline = RAGPipeline(embedder=embedder)
    pipeline.load_bundle(current)
    return pipeline.ask(question)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yt_rag.maintain",
        description="Offline index maintenance: refresh into a new versioned bundle, roll the CURRENT pointer back",
    )
    parser.add_argument(
        "--fixture",
        help="re-ingest + re-embed this transcript .txt into a NEW versioned bundle (refresh)",
    )
    parser.add_argument("--label", default="bundle", help="bundle name prefix (e.g. v1, v2)")
    parser.add_argument("--rollback", help="point CURRENT back at this bundle name (or path)")
    parser.add_argument("--list", action="store_true", help="list bundles and the current pointer")
    parser.add_argument(
        "--question", help="after refresh/rollback: query the current bundle as a smoke test"
    )
    parser.add_argument(
        "--real-embedder",
        action="store_true",
        help="rebuild with the pinned sentence-transformers model (downloads weights once; network)",
    )
    parser.add_argument("--artifacts-dir", default=None, help="override the artifacts root")
    args = parser.parse_args(argv)

    artifacts_dir = args.artifacts_dir
    try:
        if args.list:
            current = read_pointer(artifacts_dir)
            for bundle in list_bundles(artifacts_dir):
                marker = "  <- current" if current is not None and bundle == current else ""
                print(f"{bundle}{marker}")
            print(f"pointer: {pointer_path(artifacts_dir)}")
            return 0

        if not (args.fixture or args.rollback):
            parser.error("provide --fixture (refresh) or --rollback")

        if args.fixture and args.rollback:
            parser.error("--fixture and --rollback are mutually exclusive")

        embedder = None
        if args.real_embedder:
            from yt_rag.embeddings import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder()  # downloads on first use

        if args.fixture:
            out = refresh(
                args.fixture, artifacts_dir=artifacts_dir, label=args.label, embedder=embedder
            )
            print(f"refreshed: new current bundle {out}", file=sys.stderr)
        else:
            out = rollback(args.rollback, artifacts_dir=artifacts_dir, embedder=embedder)
            print(f"rolled back: current bundle is now {out}", file=sys.stderr)

        if args.question:
            result = ask_current(args.question, artifacts_dir=artifacts_dir, embedder=embedder)
            json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
            print()
        return 0
    except (BundleError, IngestError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
