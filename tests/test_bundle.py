"""Stage 3 artifact-bundle tests: manifest identity + save/load roundtrip (offline)."""

import json

import pytest

from yt_rag.bundle import (
    BUNDLE_FORMAT_VERSION,
    BundleError,
    build_manifest,
    embedder_model_id,
    load_bundle,
)
from yt_rag.config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS
from yt_rag.embeddings import HashEmbedder
from yt_rag.errors import IndexStateError
from yt_rag.pipeline import RAGPipeline


def build_offline_bundle(tmp_path, teal_fixture):
    """Ingest a fixture with the offline embedder and save a bundle."""
    pipeline = RAGPipeline(embedder=HashEmbedder(dim=384))
    pipeline.ingest_from_file(teal_fixture)
    out = tmp_path / "bundle"
    pipeline.save_bundle(out)
    return pipeline, out


def test_bundle_files_exist(tmp_path, teal_fixture):
    _, out = build_offline_bundle(tmp_path, teal_fixture)
    assert (out / "index.faiss").is_file()
    assert (out / "chunks.json").is_file()
    assert (out / "manifest.json").is_file()


def test_manifest_identity_fields(tmp_path, teal_fixture):
    _, out = build_offline_bundle(tmp_path, teal_fixture)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_format"] == BUNDLE_FORMAT_VERSION
    assert manifest["embedder"]["model_id"] == embedder_model_id(HashEmbedder(dim=384))
    assert manifest["embedder"]["dim"] == 384
    assert manifest["chunking"] == {
        "size_chars": CHUNK_SIZE_CHARS,
        "overlap_chars": CHUNK_OVERLAP_CHARS,
    }
    assert manifest["retrieval"]["top_k"] == 4
    assert "created_at" in manifest
    # manifest stores identity, never weights
    serialized = json.dumps(manifest)
    assert "weights" not in serialized and ".safetensors" not in serialized


def test_build_manifest_records_pinned_model():
    manifest = build_manifest(model_id="sentence-transformers/all-MiniLM-L6-v2", dim=384)
    assert manifest["embedder"]["model_id"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert manifest["git_commit"] is None or isinstance(manifest["git_commit"], str)


def test_bundle_roundtrip_and_query(tmp_path, teal_fixture):
    _, out = build_offline_bundle(tmp_path, teal_fixture)

    fresh = RAGPipeline(embedder=HashEmbedder(dim=384))
    manifest = fresh.load_bundle(out)
    assert manifest["embedder"]["dim"] == 384
    result = fresh.ask("how do I optimize my LinkedIn profile with ChatGPT")
    assert result["retrieved"], "reloaded bundle must retrieve"
    assert result["answer"]


def test_load_bundle_rejects_model_mismatch(tmp_path, teal_fixture):
    _, out = build_offline_bundle(tmp_path, teal_fixture)
    # The bundle was built with the hash embedder; expecting the pinned MiniLM
    # identity must fail loudly instead of silently mismatching embeddings.
    with pytest.raises(BundleError, match="model_id"):
        load_bundle(out, expected_model_id="sentence-transformers/all-MiniLM-L6-v2")


def test_load_bundle_rejects_dim_mismatch(tmp_path, teal_fixture):
    _, out = build_offline_bundle(tmp_path, teal_fixture)
    with pytest.raises(BundleError, match="dim"):
        load_bundle(out, expected_model_id=embedder_model_id(HashEmbedder(dim=384)), expected_dim=7)


def test_load_bundle_missing_manifest(tmp_path):
    (tmp_path / "notabundle").mkdir()
    with pytest.raises(BundleError, match="manifest"):
        load_bundle(tmp_path / "notabundle", expected_model_id=None)


def test_load_bundle_missing_index(tmp_path):
    d = tmp_path / "onlymanifest"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps(build_manifest(model_id="x", dim=384)), encoding="utf-8"
    )
    with pytest.raises(IndexStateError):
        load_bundle(d, expected_model_id=None)


def test_pipeline_bundle_defaults_under_artifacts_dir(tmp_path, teal_fixture):
    pipeline = RAGPipeline(embedder=HashEmbedder(dim=384), artifacts_dir=tmp_path)
    pipeline.ingest_from_file(teal_fixture)
    out = pipeline.save_bundle()
    assert out == tmp_path / "bundle"
    assert (out / "manifest.json").is_file()


def test_save_bundle_without_ingest_raises(tmp_path):
    pipeline = RAGPipeline(embedder=HashEmbedder(dim=384), artifacts_dir=tmp_path)
    with pytest.raises(RuntimeError):
        pipeline.save_bundle()
