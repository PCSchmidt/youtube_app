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
