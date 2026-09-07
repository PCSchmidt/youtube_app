"""Stage 2 eval harness tests. Fully offline: HashEmbedder + StubProvider only."""

from __future__ import annotations

import pytest

from yt_rag.eval import (
    EvalItem,
    EvalSet,
    build_embedder,
    git_commit,
    hit_rate_at_k,
    lexical_groundedness,
    load_eval_set,
    maybe_llm_notes,
    mrr_at_k,
    refusal_detected,
    run_full_eval,
    verify_eval_set,
)


def _item(**overrides) -> EvalItem:
    defaults: dict = {
        "id": "t1",
        "query": "q",
        "fixture": "game_dev_ai_paper.txt",
        "absent": False,
        "gold_span": None,
        "gold_chunk_indices": [0],
    }
    defaults.update(overrides)
    return EvalItem(**defaults)


# --- metric math ---------------------------------------------------------------


def test_hit_rate_at_k_basic():
    ranks = [1, 3, None, 5]
    assert hit_rate_at_k(ranks, k=4) == 0.5
    assert hit_rate_at_k(ranks, k=5) == 0.75
    assert hit_rate_at_k(ranks, k=1) == 0.25


def test_hit_rate_at_k_empty():
    assert hit_rate_at_k([], k=4) == 0.0


def test_mrr_at_k_basic():
    ranks = [1, 2, None, 4]
    # (1/1 + 1/2 + 0 + 1/4) / 4 = 0.4375
    assert mrr_at_k(ranks, k=4) == pytest.approx(0.4375)
    # rank beyond k counts as zero
    assert mrr_at_k(ranks, k=2) == pytest.approx(0.375)
    assert mrr_at_k([], k=4) == 0.0


# --- eval set integrity --------------------------------------------------------


def test_eval_set_is_valid_against_fixtures():
    eval_set = load_eval_set()
    assert verify_eval_set(eval_set) == []


def test_eval_set_size_in_expected_band():
    eval_set = load_eval_set()
    assert 8 <= len(eval_set.items) <= 20


def test_eval_set_chunking_matches_pinned_config():
    eval_set = load_eval_set()
    assert eval_set.chunking_at_label_time == {"chunk_size_chars": 800, "overlap_chars": 150}


def test_verify_flags_span_drift():
    real = load_eval_set()
    broken_items = [
        _item(
            id=it.id,
            query=it.query,
            fixture=it.fixture,
            absent=it.absent,
            gold_span=it.gold_span,
            gold_chunk_indices=[(i + 1) % 99 for i in it.gold_chunk_indices]
            if not it.absent
            else [],
        )
        for it in real.items
    ]
    broken = EvalSet(
        schema_version=real.schema_version,
        description=real.description,
        chunking_at_label_time=real.chunking_at_label_time,
        fixtures=real.fixtures,
        items=broken_items,
    )
    assert verify_eval_set(broken)  # at least one problem detected


# --- retrieval eval end to end (deterministic smoke, HashEmbedder) -------------


def test_hash_embedder_run_is_deterministic():
    eval_set = load_eval_set()
    a = run_full_eval("hash", eval_set=eval_set)
    b = run_full_eval("hash", eval_set=eval_set)
    assert a["retrieval"] == b["retrieval"]
    assert a["generation"] == b["generation"]
    assert [r["first_gold_rank"] for r in a["per_item"]] == [
        r["first_gold_rank"] for r in b["per_item"]
    ]


def test_hash_run_structure_and_stub_labeling():
    run = run_full_eval("hash", eval_set=load_eval_set())
    assert run["embedder"] == "HashEmbedder"
    assert run["retrieval"]["k"] == 4
    assert run["retrieval"]["n_scored"] == 12  # absent items excluded
    assert run["generation"]["provider"] == "StubProvider"
    assert run["generation"]["is_llm"] is False
    assert 0.0 <= run["retrieval"]["hit_rate_at_k"] <= 1.0
    assert 0.0 <= run["retrieval"]["mrr_at_k"] <= 1.0
    absent_rows = [r for r in run["per_item"] if r["absent"]]
    assert absent_rows and all(r["first_gold_rank"] is None for r in absent_rows)


# --- lexical generation metrics -------------------------------------------------


def test_lexical_groundedness_perfect_overlap():
    assert lexical_groundedness("the engine is neural", ["a neural engine design"]) == 1.0


def test_lexical_groundedness_no_overlap():
    assert lexical_groundedness("quantum banana", ["totally unrelated text"]) == 0.0


def test_lexical_groundedness_empty_answer():
    assert lexical_groundedness("", ["text"]) == 0.0


def test_refusal_detection():
    assert refusal_detected("I cannot find it in the transcript.")
    assert refusal_detected("That is not in the transcript.")
    assert not refusal_detected("The engine is GameNGen.")
    assert not refusal_detected("")


def test_llm_notes_skipped_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    notes = maybe_llm_notes(load_eval_set())
    assert notes["status"] == "skipped"


def test_build_embedder_unknown_name():
    with pytest.raises(ValueError):
        build_embedder("gpt4")


def test_git_commit_returns_something():
    assert isinstance(git_commit(), str) and git_commit()
