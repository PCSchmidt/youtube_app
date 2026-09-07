"""FAISS store tests: add, search, save/load roundtrip (offline)."""

import numpy as np
import pytest

from yt_rag.chunk import Chunk
from yt_rag.errors import IndexStateError
from yt_rag.vectorstore import FaissVectorStore


def make_vecs(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def chunks_for(n):
    return [
        Chunk(text=f"chunk {i}", index=i, start_char=i * 10, end_char=i * 10 + 10) for i in range(n)
    ]


def test_add_and_exact_search():
    store = FaissVectorStore(dim=8)
    vecs = make_vecs(5, 8)
    store.add(vecs, chunks_for(5))
    assert len(store) == 5
    hits = store.search(vecs[2], k=1)
    assert hits[0][1].index == 2
    assert hits[0][0] == pytest.approx(1.0, abs=1e-5)


def test_search_orders_by_similarity():
    store = FaissVectorStore(dim=8)
    vecs = make_vecs(5, 8, seed=1)
    store.add(vecs, chunks_for(5))
    hits = store.search(vecs[4], k=5)
    scores = [s for s, _ in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0][1].index == 4


def test_search_empty_store_raises():
    store = FaissVectorStore(dim=8)
    with pytest.raises(IndexStateError):
        store.search(np.zeros(8, dtype=np.float32), k=1)


def test_add_shape_mismatch_raises():
    store = FaissVectorStore(dim=8)
    with pytest.raises(ValueError):
        store.add(make_vecs(2, 8), chunks_for(3))


def test_save_load_roundtrip(tmp_path):
    store = FaissVectorStore(dim=8)
    vecs = make_vecs(5, 8, seed=2)
    store.add(vecs, chunks_for(5))
    out = store.save(tmp_path / "art")
    assert (out / "index.faiss").is_file()
    assert (out / "chunks.json").is_file()

    loaded = FaissVectorStore.load(tmp_path / "art")
    assert len(loaded) == 5
    hits_original = store.search(vecs[3], k=2)
    hits_loaded = loaded.search(vecs[3], k=2)
    assert [h[1].index for h in hits_original] == [h[1].index for h in hits_loaded]
    assert hits_original[0][0] == pytest.approx(hits_loaded[0][0], abs=1e-5)


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(IndexStateError):
        FaissVectorStore.load(tmp_path / "nothing-here")
