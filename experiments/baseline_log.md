# Baseline run log (Stage 2)

Pre-tuning baselines of the Stage 1 pipeline AS-IS. No tuning has happened:
800/150 chunking, top-k=4, IndexFlatIP cosine (inner product on L2-normalized
vectors), all-MiniLM-L6-v2 pin. Measured on `experiments/eval_set.json`
(14 hand-labeled items over the two fixture transcripts; 12 answerable + 2
absent; absent items are excluded from retrieval metrics).

Full JSON run records: `experiments/runs/`. Reproduce with `make eval`.

| Date (UTC) | Commit | Embedder | Chunk config | k | n (items / scored) | hit rate@4 | MRR@4 | Groundedness (lexical, STUB) | Run record | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-07T15:04:47+00:00 | ceb5ecf | sentence-transformers/all-MiniLM-L6-v2 (SentenceTransformerEmbedder) | 800/150 | 4 | 14 / 12 | 0.917 | 0.653 | 0.907 | `20260907T150447+0000_minilm_k4.json` | quality baseline; MiniLM weights downloaded on first use |
| 2026-09-07T15:04:33+00:00 | ceb5ecf | HashEmbedder | 800/150 | 4 | 14 / 12 | 0.917 | 0.660 | 0.905 | `20260907T150433+0000_hash_k4.json` | deterministic smoke, NOT a quality measure |

## Metric definitions

- **hit rate@4**: fraction of the 12 answerable items whose top-4 retrieved
  chunks include at least one gold chunk (a chunk whose text contains the gold
  span verbatim).
- **MRR@4**: mean reciprocal rank of the first gold chunk within the top-4.
- **Groundedness (lexical)**: mean fraction of answer content tokens present in
  the retrieved chunk text. Every number here was produced with `StubProvider`,
  an extractive stub, NOT an LLM, so groundedness is **STUB**: the stub echoes
  retrieved sentences, so high values are expected by construction.
- **Refusal on absent items**: 0.0 in both runs. StubProvider cannot say
  "cannot find it in the transcript"; measuring refusal requires an LLM provider.

## Qualitative LLM review

Not run: `OPENAI_COMPATIBLE_API_KEY` was not set on this machine at eval time.
`make eval` records qualitative LLM notes automatically when the key is present
(see `yt_rag.eval.maybe_llm_notes`). The gap is recorded here rather than faked.


## Phase 1 groundedness evaluation (2026-09-08)

The token-level lexical groundedness stub was replaced by a deterministic,
fully offline claim-level evaluator (`src/yt_rag.groundedness`), integrated
into `run_full_eval` and `make eval`. It classifies each answer as
`empty`, `evasive` (refusal phrase), `supported`, `partially_supported`, or
`unsupported`, by extracting claim sentences and checking (a) coverage of
content words against the retrieved context (>= 0.6) and (b) that every
number in a claim appears verbatim in the context (the only
fabricated-figure signal available lexically). Known `StubProvider` wrapper
lines are stripped as metadata, not scored. A committed labeled set
(`experiments/groundedness_set.json`, 9 cases: supported, partially
supported, unsupported, numeric-contradiction, missing-context, evasive,
empty) is replayed in every run record.

**What the numbers below are NOT:** these are lexical/heuristic labels, not
semantic truth. Word reuse does not make a claim true; a true paraphrase with
different words can score unsupported; there is no contradiction detection
beyond numbers/words absent from the context; question relevance is not
measured. Every generation in these runs is still `StubProvider`, an
extractive echo of retrieved chunks, so high supportedness is expected BY
CONSTRUCTION. These numbers say nothing about LLM answer quality.

| Date (UTC) | Commit | Embedder | Generation | Verdicts (12 present items) | claim groundedness | Lexical (token-level) | Labeled-set agreement | Refusal on absent | Run record |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-08T11:50:17+00:00 | a8358d5 | all-MiniLM-L6-v2 (SentenceTransformerEmbedder) | StubProvider | 12 supported / 0 partially / 0 unsupported | 1.0 | 0.907 | 9/9 | 0.0 | `20260908T115017+0000_minilm_k4.json` |
| 2026-09-08T11:50:09+00:00 | a8358d5 | HashEmbedder | StubProvider | 12 supported / 0 partially / 0 unsupported | 1.0 | 0.905 | 9/9 | 0.0 | `20260908T115009+0000_hash_k4.json` |

- **Labeled-set agreement 9/9**: the evaluator reproduces the author's
  expected verdicts on `experiments/groundedness_set.json`. This validates the
  evaluator against its documented rules; it is not an accuracy benchmark.
- **Refusal on absent items is still 0.0**: `StubProvider` echoes retrieved
  text even when the answer is absent, and those echoes score "supported"
  because they are built from retrieved chunks. Measuring refusal requires an
  LLM provider.
- **Qualitative LLM review: still open.** `OPENAI_COMPATIBLE_API_KEY` was not
  set at eval time; `make eval --llm-notes` records qualitative notes when the
  key is present. The Stage 2 generation-metrics box stays partially open
  until that runs.
- Reproduce with `make eval` (offline default embedder: `python -m yt_rag.eval
  --embedder hash`); the evaluator and labeled-set replay are fully offline.
