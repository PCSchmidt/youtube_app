"""Groundedness evaluator tests (Phase 1). Fully offline and deterministic.

The labeled set (experiments/groundedness_set.json) must agree with the
evaluator 100% — it checks the evaluator, not answer truth.
"""

from __future__ import annotations

import pytest

from yt_rag.groundedness import (
    DEFAULT_GROUNDEDNESS_SET,
    GROUNDEDNESS_LIMITATIONS,
    SUPPORT_THRESHOLD,
    classify_claim,
    evaluate_groundedness,
    extract_claims,
    load_groundedness_set,
    run_groundedness_set,
)

CTX = ["220 characters is the max limit for your LinkedIn headline"]


# --- claim extraction -----------------------------------------------------------


def test_extract_claims_splits_sentences():
    claims = extract_claims("LinkedIn headline limit is 220 characters. It also has keywords.")
    assert len(claims) == 2
    assert claims[0] == "LinkedIn headline limit is 220 characters."


def test_extract_claims_strips_stub_wrapper_metadata():
    answer = (
        "[stub answer, grounded in 4 retrieved chunk(s)]\n"
        "Relevant transcript passage (chunk 1):\n"
        "the graphics are simulated at 20 frames per second."
    )
    claims = extract_claims(answer)
    assert claims == ["the graphics are simulated at 20 frames per second."]


def test_extract_claims_drops_contentless_sentences():
    # every token is a stopword -> nothing checkable
    assert extract_claims("Is it? Is that it? So be it.") == []
    assert extract_claims("") == []


# --- claim classification -------------------------------------------------------


def test_classify_claim_supported():
    v = classify_claim("LinkedIn headline limit is 220 characters", CTX[0].lower())
    assert v.verdict == "supported"
    assert v.coverage == 1.0
    assert v.unsupported_tokens == ()


def test_classify_claim_missing_number_is_unsupported_even_at_threshold():
    # 4/5 tokens covered (0.8 >= threshold) but the number 500 is fabricated.
    assert SUPPORT_THRESHOLD <= 0.8
    v = classify_claim("LinkedIn headline limit is 500 characters", CTX[0].lower())
    assert v.verdict == "unsupported"
    assert "500" in v.unsupported_tokens


def test_classify_claim_low_coverage_is_unsupported():
    v = classify_claim("OpenAI released GPT-5 in Seattle", CTX[0].lower())
    assert v.verdict == "unsupported"
    assert set(v.unsupported_tokens) >= {"openai", "seattle"}


# --- answer-level evaluation ----------------------------------------------------


def test_empty_answer_is_empty():
    r = evaluate_groundedness("   ", CTX)
    assert r.verdict == "empty"
    assert r.n_claims == 0
    assert r.claim_groundedness is None


def test_refusal_answer_is_evasive():
    r = evaluate_groundedness("I cannot find it in the transcript.", CTX)
    assert r.verdict == "evasive"
    assert r.evasion_detected is True


def test_supported_answer():
    r = evaluate_groundedness("LinkedIn headline limit is 220 characters.", CTX)
    assert r.verdict == "supported"
    assert (r.n_claims, r.n_supported, r.n_unsupported) == (1, 1, 0)
    assert r.claim_groundedness == 1.0


def test_partially_supported_answer():
    answer = "LinkedIn headline limit is 220 characters. The same applies on every social network."
    r = evaluate_groundedness(answer, CTX)
    assert r.verdict == "partially_supported"
    assert (r.n_claims, r.n_supported, r.n_unsupported) == (2, 1, 1)
    assert r.claim_groundedness == pytest.approx(0.5)


def test_unsupported_answer():
    r = evaluate_groundedness("OpenAI released GPT-5 with video reasoning in Seattle.", CTX)
    assert r.verdict == "unsupported"
    assert r.claim_groundedness == 0.0


def test_missing_context_flags_all_claims_unsupported():
    r = evaluate_groundedness("The engine runs at 20 frames per second.", [])
    assert r.verdict == "unsupported"
    assert r.n_unsupported == r.n_claims == 1


def test_nonempty_answer_without_claims_is_unsupported():
    # only stopwords -> no extractable claims, nothing checkable
    r = evaluate_groundedness("It is.", CTX)
    assert r.verdict == "unsupported"
    assert r.claim_groundedness is None


def test_evaluator_is_deterministic():
    answer = "LinkedIn headline limit is 220 characters. The same applies on every social network."
    a = evaluate_groundedness(answer, CTX).to_dict()
    b = evaluate_groundedness(answer, CTX).to_dict()
    assert a == b


def test_result_to_dict_is_json_serializable():
    import json

    r = evaluate_groundedness("LinkedIn headline limit is 220 characters.", CTX)
    assert json.loads(json.dumps(r.to_dict()))["verdict"] == "supported"


# --- labeled groundedness set ---------------------------------------------------


def test_groundedness_set_agreement_is_full():
    summary = run_groundedness_set()
    assert summary["n_cases"] >= 6
    assert summary["n_agree"] == summary["n_cases"]
    assert summary["agreement"] == 1.0


def test_groundedness_set_covers_all_required_categories():
    data = load_groundedness_set()
    categories = {c["category"] for c in data["cases"]}
    assert {"supported", "unsupported", "evasive", "empty"} <= categories
    verdicts = {c["expected_verdict"] for c in data["cases"]}
    assert {"supported", "partially_supported", "unsupported", "evasive", "empty"} <= verdicts


def test_groundedness_set_every_case_is_replayed_by_evaluator():
    # redundant with agreement, but explicit: expected verdict == evaluator verdict
    data = load_groundedness_set()
    for case in data["cases"]:
        r = evaluate_groundedness(case["answer"], case["context"])
        assert r.verdict == case["expected_verdict"], case["id"]


def test_groundedness_set_has_provenance_description():
    data = load_groundedness_set()
    assert "NOT semantic truth" in data["description"]
    assert "tests/fixtures/" in data["description"]


def test_limitations_string_states_not_semantic_truth():
    assert "NOT semantic truth" in GROUNDEDNESS_LIMITATIONS
    assert "contradiction" in GROUNDEDNESS_LIMITATIONS


# --- integration with the eval harness ------------------------------------------


def test_run_full_eval_includes_groundedness_block():
    from yt_rag.eval import run_full_eval

    run = run_full_eval("hash", eval_set=None)
    gen = run["generation"]
    assert set(gen["groundedness_verdict_counts"]) <= {
        "supported",
        "partially_supported",
        "unsupported",
        "empty",
        "evasive",
    }
    assert sum(gen["groundedness_verdict_counts"].values()) == 12  # present items only
    assert gen["claim_groundedness_mean"] is not None
    assert run["groundedness_set"]["agreement"] == 1.0
    assert run["groundedness_limitations"] == GROUNDEDNESS_LIMITATIONS
    # StubProvider echoes retrieved text, so verdicts must be non-negative claims
    for row in run["per_item"]:
        assert row["groundedness"]["verdict"] in {
            "supported",
            "partially_supported",
            "unsupported",
            "empty",
            "evasive",
        }
    assert DEFAULT_GROUNDEDNESS_SET.exists()
