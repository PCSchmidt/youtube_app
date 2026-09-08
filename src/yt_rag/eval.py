"""Evaluation harness (Stage 2): retrieval + generation metrics over a labeled set.

Measures the Stage 1 pipeline AS-IS (no tuning): hit rate@k and MRR@k against a
small hand-labeled eval set, plus deterministic groundedness evaluation of the
generated answer against the retrieved context (see yt_rag.groundedness:
claim-level supported/unsupported verdicts, empty/evasive detection, and a
token-level lexical overlap proxy) and a refusal check for questions whose
answer is absent from the transcript. Everything runs offline with the
deterministic HashEmbedder + StubProvider; the pinned MiniLM model is used only
when explicitly requested (via `make eval` or
``python -m yt_rag.eval --embedder minilm``), which may download weights on
first use.

The default provider is StubProvider, an extractive stub, NOT an LLM: any
groundedness number measured with it says nothing about LLM answer quality.
And in all cases the groundedness labels are LEXICAL HEURISTICS, NOT semantic
truth (see yt_rag.groundedness.GROUNDEDNESS_LIMITATIONS, embedded in every run
record).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from yt_rag.chunk import chunk_text
from yt_rag.config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, DEFAULT_TOP_K, REPO_ROOT
from yt_rag.embeddings import EmbeddingProvider, HashEmbedder
from yt_rag.generation import GenerationProvider, StubProvider
from yt_rag.groundedness import (
    EVASIVE_PHRASES,
    GROUNDEDNESS_LIMITATIONS,
    content_tokens,
    evaluate_groundedness,
    run_groundedness_set,
)
from yt_rag.ingest import load_transcript_file
from yt_rag.pipeline import RAGPipeline

DEFAULT_EVAL_SET = REPO_ROOT / "experiments" / "eval_set.json"
DEFAULT_RUNS_DIR = REPO_ROOT / "experiments" / "runs"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

# Backwards-compatible alias for the refusal-phrase list (now defined once in
# yt_rag.groundedness alongside the full evaluator).
_REFUSAL_PHRASES = EVASIVE_PHRASES


@dataclass(frozen=True)
class EvalItem:
    """One hand-labeled eval item."""

    id: str
    query: str
    fixture: str
    absent: bool
    gold_span: str | None
    gold_chunk_indices: list[int]


@dataclass(frozen=True)
class EvalSet:
    """The labeled eval set plus its fixture provenance metadata."""

    schema_version: int
    description: str
    chunking_at_label_time: dict
    fixtures: dict
    items: list[EvalItem]


def load_eval_set(path: str | Path | None = None) -> EvalSet:
    """Load the labeled eval set from its versioned JSON file."""
    p = Path(path) if path else DEFAULT_EVAL_SET
    raw = json.loads(p.read_text(encoding="utf-8"))
    items = [EvalItem(**item) for item in raw["items"]]
    return EvalSet(
        schema_version=int(raw["schema_version"]),
        description=raw["description"],
        chunking_at_label_time=raw["chunking_at_label_time"],
        fixtures=raw["fixtures"],
        items=items,
    )


def fixture_text(name: str) -> str:
    """Load a fixture transcript by its filename in tests/fixtures/."""
    return load_transcript_file(FIXTURES_DIR / name).text


def gold_chunks_for_span(text: str, span: str) -> list[int]:
    """Chunk indices (pinned 800/150 chunker) whose text contains the span."""
    return [c.index for c in chunk_text(text) if span in c.text]


def verify_eval_set(eval_set: EvalSet) -> list[str]:
    """Check every gold span and chunk index against the fixture text.

    Returns a list of problems (empty means the eval set is internally
    consistent with the fixtures).
    """
    problems: list[str] = []
    for item in eval_set.items:
        text = fixture_text(item.fixture)
        if item.absent:
            if item.gold_span is not None or item.gold_chunk_indices:
                problems.append(f"{item.id}: absent item must have no gold span/chunks")
            continue
        if item.gold_span is None:
            problems.append(f"{item.id}: present item is missing a gold span")
            continue
        if item.gold_span not in text:
            problems.append(f"{item.id}: gold span not found verbatim in {item.fixture}")
        expected = gold_chunks_for_span(text, item.gold_span)
        if sorted(item.gold_chunk_indices) != expected:
            problems.append(
                f"{item.id}: gold_chunk_indices {item.gold_chunk_indices} != "
                f"recomputed {expected} (chunking config changed?)"
            )
        n_chunks = len(chunk_text(text))
        for idx in item.gold_chunk_indices:
            if not 0 <= idx < n_chunks:
                problems.append(f"{item.id}: gold chunk {idx} out of range (0..{n_chunks - 1})")
    return problems


# --- lexical generation metrics ------------------------------------------------


def lexical_groundedness(answer: str, context_texts: list[str]) -> float:
    """Fraction of answer content tokens that appear in the context texts.

    Cheap TOKEN-level lexical proxy, kept alongside the claim-level evaluator
    in yt_rag.groundedness. 1.0 means every content word in the answer
    appears somewhere in retrieved chunk text; lower means the answer
    contains unsupported tokens. NOT an LLM judge, and NOT semantic truth:
    a paraphrase with different words scores low, and word reuse does not
    make a claim true.
    """
    tokens = content_tokens(answer)
    if not tokens:
        return 0.0
    context = " ".join(context_texts).lower()
    supported = sum(1 for t in tokens if t in context)
    return supported / len(tokens)


def refusal_detected(answer: str) -> bool:
    """True if the answer indicates the transcript does not contain the answer.

    Alias for yt_rag.groundedness.evasion_detected (same phrase list).
    """
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)


# --- retrieval metrics ---------------------------------------------------------


def hit_rate_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of queries whose first gold chunk appears in the top-k."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr_at_k(ranks: list[int | None], k: int) -> float:
    """Mean reciprocal rank of the first gold chunk within the top-k."""
    if not ranks:
        return 0.0
    recip = [1.0 / r if r is not None and r <= k else 0.0 for r in ranks]
    return sum(recip) / len(recip)


def _pipeline_for(
    fixture: str, embedder: EmbeddingProvider, provider: GenerationProvider, k: int
) -> RAGPipeline:
    pipeline = RAGPipeline(embedder=embedder, provider=provider, top_k=k)
    pipeline.ingest_from_file(FIXTURES_DIR / fixture)
    return pipeline


def _verdict_counts(rows: list[dict]) -> dict[str, int]:
    """Count answer-level groundedness verdicts over the given rows."""
    counts: dict[str, int] = {}
    for r in rows:
        v = r["groundedness"]["verdict"]
        counts[v] = counts.get(v, 0) + 1
    return dict(sorted(counts.items()))


def _claim_groundedness_mean(rows: list[dict]) -> float | None:
    """Mean claim_groundedness over rows that have checkable claims."""
    values = [r["groundedness"]["claim_groundedness"] for r in rows]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def run_retrieval_eval(
    eval_set: EvalSet,
    embedder: EmbeddingProvider,
    k: int = DEFAULT_TOP_K,
    provider: GenerationProvider | None = None,
) -> dict:
    """Retrieve top-k for every eval item and score hit rate / MRR / groundedness.

    Absent items are excluded from retrieval metrics; they are only checked for
    a cannot-find-it refusal. With the default StubProvider the groundedness
    and refusal numbers are STUB numbers, not LLM quality.
    """
    provider = provider or StubProvider()
    pipelines: dict[str, RAGPipeline] = {}
    rows: list[dict] = []
    for item in eval_set.items:
        if item.fixture not in pipelines:
            pipelines[item.fixture] = _pipeline_for(item.fixture, embedder, provider, k)
        retrieved = pipelines[item.fixture].retrieve(item.query)
        retrieved_indices = [r.chunk.index for r in retrieved]
        retrieved_texts = [r.chunk.text for r in retrieved]
        first_gold_rank: int | None = None
        if not item.absent:
            for rank, idx in enumerate(retrieved_indices, start=1):
                if idx in item.gold_chunk_indices:
                    first_gold_rank = rank
                    break
        answer = provider.generate(item.query, retrieved)
        groundedness = evaluate_groundedness(answer, retrieved_texts)
        row = {
            "id": item.id,
            "query": item.query,
            "fixture": item.fixture,
            "absent": item.absent,
            "gold_chunk_indices": item.gold_chunk_indices,
            "retrieved_chunk_indices": retrieved_indices,
            "first_gold_rank": first_gold_rank,
            "answer": answer,
            "groundedness_lexical": lexical_groundedness(answer, retrieved_texts),
            "groundedness": groundedness.to_dict(),
            "refusal_detected": groundedness.evasion_detected,
        }
        rows.append(row)

    present = [r for r in rows if not r["absent"]]
    absent = [r for r in rows if r["absent"]]
    ranks = [r["first_gold_rank"] for r in present]
    return {
        "rows": rows,
        "retrieval": {
            "k": k,
            "n_scored": len(present),
            "hit_rate_at_k": hit_rate_at_k(ranks, k),
            "mrr_at_k": mrr_at_k(ranks, k),
        },
        "generation": {
            "provider": type(provider).__name__,
            "is_llm": not isinstance(provider, StubProvider),
            "groundedness_lexical_mean": (
                sum(r["groundedness_lexical"] for r in present) / len(present)
            )
            if present
            else 0.0,
            "groundedness_verdict_counts": _verdict_counts(present),
            "claim_groundedness_mean": _claim_groundedness_mean(present),
            "refusal_rate_on_absent": (
                sum(1 for r in absent if r["refusal_detected"]) / len(absent)
            )
            if absent
            else None,
        },
    }


# --- run logging ---------------------------------------------------------------


def git_commit() -> str:
    """Short commit hash of the working tree HEAD (or 'unknown' outside git)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def build_embedder(name: str) -> EmbeddingProvider:
    """Map an eval embedder name to its provider. 'minilm' downloads weights."""
    name = name.lower()
    if name == "hash":
        return HashEmbedder()
    if name == "minilm":
        from yt_rag.embeddings import SentenceTransformerEmbedder  # lazy: heavy

        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder {name!r} (use 'hash' or 'minilm')")


def maybe_llm_notes(
    eval_set: EvalSet,
    model: str | None = None,
    max_items: int = 3,
) -> dict:
    """Optional qualitative notes from a real LLM. Skipped without an API key.

    Never called by the test suite; only by the eval CLI. A missing key is
    recorded, not raised, so offline runs stay green.
    """
    if not os.environ.get("OPENAI_COMPATIBLE_API_KEY"):
        return {
            "status": "skipped",
            "reason": "OPENAI_COMPATIBLE_API_KEY not set; no qualitative LLM review",
        }
    from yt_rag.generation import OpenAICompatibleProvider

    llm = OpenAICompatibleProvider(
        model=model or os.environ.get("YT_RAG_EVAL_LLM_MODEL", "openai/gpt-4o-mini")
    )
    notes = []
    for item in eval_set.items[:max_items]:
        pipeline = _pipeline_for(item.fixture, build_embedder("minilm"), llm, DEFAULT_TOP_K)
        result = pipeline.ask(item.query)
        notes.append({"id": item.id, "query": item.query, "answer": result["answer"]})
    return {"status": "ok", "provider": type(llm).__name__, "model": llm.model, "notes": notes}


def run_full_eval(
    embedder_name: str,
    k: int = DEFAULT_TOP_K,
    eval_set: EvalSet | None = None,
    with_llm_notes: bool = False,
) -> dict:
    """One complete eval run of the measured pipeline, ready to log as JSON."""
    eval_set = eval_set or load_eval_set()
    problems = verify_eval_set(eval_set)
    if problems:
        raise ValueError(f"eval set failed verification: {problems}")
    embedder = build_embedder(embedder_name)
    result = run_retrieval_eval(eval_set, embedder, k=k)
    run = {
        "schema_version": 1,
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "embedder": type(embedder).__name__,
        "embedder_requested": embedder_name,
        "chunk_config": {
            "chunk_size_chars": CHUNK_SIZE_CHARS,
            "overlap_chars": CHUNK_OVERLAP_CHARS,
        },
        "top_k": k,
        "eval_set_n": len(eval_set.items),
        "retrieval": result["retrieval"],
        "generation": result["generation"],
        "groundedness_set": run_groundedness_set(),
        "groundedness_limitations": GROUNDEDNESS_LIMITATIONS,
        "llm_qualitative_notes": maybe_llm_notes(eval_set) if with_llm_notes else None,
        "per_item": result["rows"],
    }
    return run


def save_run(run: dict, out_dir: str | Path | None = None) -> Path:
    """Write one run record to experiments/runs/ and return its path."""
    out = Path(out_dir) if out_dir else DEFAULT_RUNS_DIR
    out.mkdir(parents=True, exist_ok=True)
    stamp = run["date_utc"].replace(":", "").replace("-", "")
    name = f"{stamp}_{run['embedder_requested']}_k{run['top_k']}.json"
    path = out / name
    path.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI: run the eval and write run records. `make eval` calls this.

    This is the only yt_rag entry point allowed to touch the network (MiniLM
    weight download). It is NOT part of `make test`.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="yt_rag.eval", description="run the Stage 2 eval (may download model weights)"
    )
    parser.add_argument("--embedder", default="both", choices=["hash", "minilm", "both"])
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--eval-set", default=None, help="path to eval_set.json")
    parser.add_argument("--out-dir", default=None, help="run record output directory")
    parser.add_argument(
        "--llm-notes",
        action="store_true",
        help="add qualitative LLM notes if OPENAI_COMPATIBLE_API_KEY is set",
    )
    args = parser.parse_args(argv)

    names = ["hash", "minilm"] if args.embedder == "both" else [args.embedder]
    eval_set = load_eval_set(args.eval_set)
    failures = 0
    for name in names:
        try:
            run = run_full_eval(name, k=args.k, eval_set=eval_set, with_llm_notes=args.llm_notes)
            path = save_run(run, args.out_dir)
            ret = run["retrieval"]
            gen = run["generation"]
            print(
                f"[{run['embedder']}] n={run['eval_set_n']} k={ret['k']} "
                f"hit_rate@{ret['k']}={ret['hit_rate_at_k']:.3f} "
                f"mrr@{ret['k']}={ret['mrr_at_k']:.3f} "
                f"groundedness_lex={gen['groundedness_lexical_mean']:.3f} "
                f"claims={gen['claim_groundedness_mean']} "
                f"verdicts={gen['groundedness_verdict_counts']} "
                f"(provider={gen['provider']})"
            )
            print(
                f"  groundedness_set agreement={run['groundedness_set']['agreement']} "
                f"({run['groundedness_set']['n_agree']}/{run['groundedness_set']['n_cases']})"
            )
            print(f"  run record: {path}")
        except Exception as exc:  # noqa: BLE001 - record ANY failure honestly, then fail
            failures += 1
            failed = {
                "schema_version": 1,
                "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "git_commit": git_commit(),
                "embedder_requested": name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            out = Path(args.out_dir) if args.out_dir else DEFAULT_RUNS_DIR
            out.mkdir(parents=True, exist_ok=True)
            stamp = failed["date_utc"].replace(":", "").replace("-", "")
            fpath = out / f"{stamp}_{name}_failed.json"
            fpath.write_text(
                json.dumps(failed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[{name}] FAILED: {exc} (recorded in {fpath})", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
