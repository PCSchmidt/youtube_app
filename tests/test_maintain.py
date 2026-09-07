"""Stage 6 maintain tests: offline refresh, pointer rollback, identity guard.

All paths are tmp dirs; bundles stay gitignored in real runs. HashEmbedder +
StubProvider + committed fixtures only — no network, no model downloads.
"""

import json

import pytest

from yt_rag.bundle import BundleError, embedder_model_id
from yt_rag.embeddings import HashEmbedder
from yt_rag.maintain import (
    list_bundles,
    read_pointer,
    refresh,
    rollback,
)
from yt_rag.pipeline import RAGPipeline


class ForeignEmbedder(HashEmbedder):
    """A hash embedder with a different identity, to force BundleError."""

    model_id = "foreign-embedder:v9"


def snapshot(directory):
    """Bytes of every file in a bundle, for prove-it-did-not-change checks."""
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def test_refresh_writes_new_versioned_bundle_and_moves_pointer(tmp_path, teal_fixture):
    arts = tmp_path / "artifacts"
    out = refresh(teal_fixture, artifacts_dir=arts, label="v1")
    assert out.parent == arts and out.is_dir()
    assert (out / "manifest.json").is_file() and (out / "index.faiss").is_file()
    assert read_pointer(arts) == out
    # versioned name, not a fixed directory
    assert out.name.startswith("v1-")


def test_refresh_never_overwrites_previous_bundle(tmp_path, teal_fixture, fixture_dir):
    arts = tmp_path / "artifacts"
    v1 = refresh(teal_fixture, artifacts_dir=arts, label="v")
    before = snapshot(v1)
    v2 = refresh(fixture_dir / "game_dev_ai_paper.txt", artifacts_dir=arts, label="v")
    assert v2 != v1 and snapshot(v1) == before, "previous bundle must be untouched"
    assert read_pointer(arts) == v2
    assert list_bundles(arts) == [v1, v2]


def test_rollback_restores_previous_manifest_and_query_works(tmp_path, teal_fixture, fixture_dir):
    arts = tmp_path / "artifacts"
    v1 = refresh(teal_fixture, artifacts_dir=arts, label="v1")
    refresh(fixture_dir / "game_dev_ai_paper.txt", artifacts_dir=arts, label="v2")
    back = rollback("v1", artifacts_dir=arts)
    assert back == v1 and read_pointer(arts) == v1

    fresh = RAGPipeline(embedder=HashEmbedder(dim=384))
    manifest = fresh.load_bundle(back)
    original = json.loads((v1 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["embedder"] == original["embedder"]
    result = fresh.ask("how do I optimize my LinkedIn profile with ChatGPT")
    assert result["retrieved"] and result["answer"]


def test_rollback_by_path(tmp_path, teal_fixture, fixture_dir):
    arts = tmp_path / "artifacts"
    v1 = refresh(teal_fixture, artifacts_dir=arts, label="v1")
    refresh(fixture_dir / "game_dev_ai_paper.txt", artifacts_dir=arts, label="v2")
    assert rollback(v1, artifacts_dir=arts) == v1
    assert read_pointer(arts) == v1


def test_rollback_identity_mismatch_leaves_pointer_untouched(tmp_path, teal_fixture):
    arts = tmp_path / "artifacts"
    v1 = refresh(teal_fixture, artifacts_dir=arts, label="v1", embedder=ForeignEmbedder(dim=384))
    assert read_pointer(arts) == v1
    # The hash embedder's identity does not match the bundle's manifest.
    with pytest.raises(BundleError, match="model_id"):
        rollback("v1", artifacts_dir=arts, embedder=HashEmbedder(dim=384))
    assert read_pointer(arts) == v1, "failed validation must not move the pointer"


def test_rollback_missing_bundle_leaves_pointer_untouched(tmp_path, teal_fixture):
    arts = tmp_path / "artifacts"
    v1 = refresh(teal_fixture, artifacts_dir=arts, label="v1")
    with pytest.raises(BundleError):
        rollback("nope-1985", artifacts_dir=arts)
    assert read_pointer(arts) == v1


def test_refresh_records_refreshing_embedder_identity(tmp_path, teal_fixture):
    arts = tmp_path / "artifacts"
    out = refresh(teal_fixture, artifacts_dir=arts, label="v1", embedder=ForeignEmbedder(dim=384))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["embedder"]["model_id"] == embedder_model_id(ForeignEmbedder(dim=384))


def test_read_pointer_unset_and_missing_bundle(tmp_path):
    assert read_pointer(tmp_path) is None
    (tmp_path / "CURRENT").write_text("ghost\n", encoding="utf-8")
    with pytest.raises(BundleError):
        read_pointer(tmp_path)
