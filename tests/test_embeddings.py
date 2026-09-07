"""Embedding tests: determinism, normalization, dimension (offline HashEmbedder)."""

import numpy as np

from yt_rag.embeddings import HashEmbedder


def test_deterministic():
    a = HashEmbedder(dim=64).embed(["the quick brown fox", "jumps"])
    b = HashEmbedder(dim=64).embed(["the quick brown fox", "jumps"])
    assert np.array_equal(a, b)


def test_normalized_unit_vectors():
    v = HashEmbedder(dim=64).embed(["a b c", "different words entirely"])
    assert v.shape == (2, 64)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_identical_text_identical_row():
    v = HashEmbedder(dim=64).embed(["same text", "same text"])
    assert np.allclose(v[0], v[1])


def test_dim_configurable():
    assert HashEmbedder(dim=128).embed(["x"]).shape == (1, 128)
