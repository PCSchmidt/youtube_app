"""FastAPI endpoint smoke tests via TestClient (offline, injected stubs)."""

import importlib

import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from yt_rag.app import create_app
from yt_rag.embeddings import HashEmbedder
from yt_rag.generation import StubProvider


@pytest.fixture(scope="module")
def client(teal_fixture):
    app = create_app(embedder=HashEmbedder(dim=384), provider=StubProvider())
    with TestClient(app) as c:
        c.teal_fixture = str(teal_fixture)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_with_file(client):
    r = client.post(
        "/chat",
        json={"file": client.teal_fixture, "question": "how to optimize a linkedin profile"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert 0 < len(body["retrieved"]) <= 4
    assert "chunk_index" in body["retrieved"][0]


def test_chat_without_source_is_422(client):
    r = client.post("/chat", json={"question": "hi"})
    assert r.status_code == 422


def test_chat_bad_url_is_422(client):
    r = client.post("/chat", json={"url": "not a url", "question": "hi"})
    assert r.status_code == 422


def test_default_app_module_importable():
    import yt_rag.app as app_module

    importlib.reload(app_module)
    assert app_module.app is not None
