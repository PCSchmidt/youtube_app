"""Shared offline test fixtures.

Tests use cached transcript .txt fixtures (originally salvaged from the
pre-refactor app), the deterministic HashEmbedder, and the StubProvider so the whole
suite runs with no network, no API keys, and no model downloads.
"""

from pathlib import Path

import pytest

from yt_rag.embeddings import HashEmbedder
from yt_rag.generation import StubProvider

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def teal_fixture() -> Path:
    return FIXTURES / "teal_chatgpt_linkedin.txt"


@pytest.fixture()
def offline_embedder() -> HashEmbedder:
    return HashEmbedder(dim=384)


@pytest.fixture()
def stub_provider() -> StubProvider:
    return StubProvider()
