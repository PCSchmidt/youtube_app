"""Stage 0 smoke tests. No pillar functionality is tested here yet."""

import sys

import yt_rag


def test_python_version():
    assert sys.version_info >= (3, 11)


def test_import_exposes_version():
    assert isinstance(yt_rag.__version__, str)
    assert yt_rag.__version__ == "0.1.0"
