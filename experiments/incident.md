# Incident 2026-09-07T21:20Z — bad refresh made the wrong transcript current; rolled back to v1

**Simulated, fixture-based, and fully executed.** Every command and output below was
actually run on branch `stage6` (commit `0e9f6fd`, Windows Git Bash, offline:
HashEmbedder + StubProvider + committed fixtures, no network, no keys). This is a
rehearsal of the maintain loop, not a production outage: there is no production service.
The `StubProvider` answers are extractive echoes of retrieved chunks by construction
(Stage 2 groundedness is a stub); "grounded" below means the answer text is built from
chunks of the correct transcript, nothing stronger.

## Timeline and detection

1. **21:20:22 — refresh v1 (healthy).** Rebuilt the index from the LinkedIn fixture
   (`teal_chatgpt_linkedin.txt`) into a new versioned bundle; pointer moved to it.
2. **21:20:27 — refresh v2 (the incident).** A "content refresh" rebuilt the index from
   the **wrong** fixture (`game_dev_ai_paper.txt`) and the pointer moved to v2. Refresh
   succeeds whenever the bundle is loadable with the current embedder identity — the
   Stage 3 identity check validates embedder model_id/dim, not *which transcript* went
   in. Wrong-source refreshes are therefore a silent failure mode.
3. **Detection via the Stage 5 proxy signal.** The smoke query (`--question`) against v2
   dropped the top cosine score from **0.257 to 0.041** and the answer switched from
   LinkedIn-profile chunks to Doom/Genie game-AI chunks. The Stage 5 observability note
   says exactly this is the signal: a nonzero retrieval with a low `top_score` means the
   vector search found nothing similar (a proxy, not a quality measure).
4. **21:20:49+ — rollback to v1.** Pointer moved back after the Stage 3 identity
   validation passed; the same smoke query returned the original grounded answer with
   identical scores (deterministic embedder).
5. **Guard check — a mismatched bundle cannot become current.** A rollback attempt
   against a bundle built with a different embedder identity raised `BundleError` and the
   pointer was left untouched.

## Commands and outputs (verbatim)

All commands run from the repo root via Git Bash.

### 1. Refresh v1 from the correct fixture

```
$ PYTHONUTF8=1 python -m yt_rag.maintain \
    --fixture tests/fixtures/teal_chatgpt_linkedin.txt --label v1 \
    --question "how do I optimize my LinkedIn profile with ChatGPT"
refreshed: new current bundle C:\...\youtube_app\artifacts\v1-20260907T212022Z
```

JSON answer (trimmed to the key fields — full output is the ChatResponse shape):

```json
{
  "question": "how do I optimize my LinkedIn profile with ChatGPT",
  "answer": "[stub answer, grounded in 4 retrieved chunk(s)]\nRelevant transcript passage (chunk 0):\nwhat's going on everybody hope your day is going well Liam here from the teal content team and in this video I want to go over four ways you can use Chad GPT to optimize your LinkedIn profile ...",
  "retrieved": [
    {"chunk_index": 0, "score": 0.2568915784358978, "text": "what's going on everybody ... optimize your LinkedIn profile ..."},
    {"chunk_index": 9, "score": 0.10913940519094467, "text": "to experience we're going to hit the pencil to edit ..."},
    {"chunk_index": 11, "score": 0.10075854510068893, "text": "to thank them for their time and consideration ..."},
    {"chunk_index": 4, "score": 0.08113206177949905, "text": "is looking stronger let's also go in our resume ..."}
  ]
}
```

### 2. The bad refresh: rebuild v2 from the wrong fixture

```
$ PYTHONUTF8=1 python -m yt_rag.maintain \
    --fixture tests/fixtures/game_dev_ai_paper.txt --label v2 \
    --question "how do I optimize my LinkedIn profile with ChatGPT"
refreshed: new current bundle C:\...\youtube_app\artifacts\v2-20260907T212027Z
```

Same question, now served from the wrong transcript — retrieval still returns 4 chunks
(it always returns top-k when the index is non-empty), but the scores collapse and the
content is unrelated:

```json
{
  "question": "how do I optimize my LinkedIn profile with ChatGPT",
  "answer": "[stub answer, grounded in 4 retrieved chunk(s)]\nRelevant transcript passage (chunk 3):\nof new programmers currently he's working on artificial general intelligence at Keen Technologies ...",
  "retrieved": [
    {"chunk_index": 3, "score": 0.04126562923192978, "text": "of new programmers ... Doom is just a bunch of 2D Sprites ..."},
    {"chunk_index": 4, "score": 0.020286019891500473, "text": "but it's also been shipping some amazing Tech ..."},
    {"chunk_index": 1, "score": 0.02021130360662937, "text": "GTA 7 comes out in 2042 ..."},
    {"chunk_index": 5, "score": 0.019407417625188828, "text": "based on the actions taken by the player ..."}
  ]
}
```

Detection summary: **top_score 0.257 -> 0.041** (Stage 5 empty-result/top-score proxy
fires), and the answer content is from the wrong transcript. In the HTTP service this is
the same signal in the `/chat` log line (`top_score`) and `/metrics`
(`mean_top_score`); the requests themselves return 200.

### 3. Pointer state before rollback

```
$ python -m yt_rag.maintain --list
C:\...\youtube_app\artifacts\v1-20260907T212022Z
C:\...\youtube_app\artifacts\v2-20260907T212027Z  <- current
pointer: C:\...\youtube_app\artifacts\CURRENT

$ cat artifacts/CURRENT
v2-20260907T212027Z
```

### 4. Rollback to v1 and re-query

```
$ PYTHONUTF8=1 python -m yt_rag.maintain --rollback v1 \
    --question "how do I optimize my LinkedIn profile with ChatGPT"
rolled back: current bundle is now C:\...\youtube_app\artifacts\v1-20260907T212022Z
```

```json
{
  "question": "how do I optimize my LinkedIn profile with ChatGPT",
  "answer": "[stub answer, grounded in 4 retrieved chunk(s)]\nRelevant transcript passage (chunk 0):\nwhat's going on everybody hope your day is going well Liam here from the teal content team ... optimize your LinkedIn profile ...",
  "retrieved": [
    {"chunk_index": 0, "score": 0.2568915784358978, "text": "what's going on everybody ... optimize your LinkedIn profile ..."},
    ...same four chunks and scores as step 1...
  ]
}
```

The rollback ran the Stage 3 `load_bundle` identity validation (embedder `model_id` +
`dim` against the manifest) before moving the pointer; `ask_current` loads through the
same validation. The query is grounded in the correct transcript again, with bit-identical
scores (HashEmbedder is deterministic).

### 5. Guard check: a failed identity validation does not move the pointer

A bundle built with a *different* embedder identity must never become current via
rollback. Executable check (contents of a small script; run with
`python <script>.py` from the repo root — `python -c` one-liners cannot define a class
cleanly on all shells):

```python
from yt_rag.embeddings import HashEmbedder
from yt_rag.bundle import BundleError
from yt_rag import maintain


class ForeignEmbedder(HashEmbedder):
    model_id = "foreign-embedder:v9"


b = maintain.refresh("tests/fixtures/teal_chatgpt_linkedin.txt", label="v3", embedder=ForeignEmbedder(dim=384))
before = maintain.read_pointer()
try:
    maintain.rollback("v3", embedder=HashEmbedder(dim=384))
except BundleError as e:
    print("BundleError:", e)
print("pointer before rollback attempt:", before.name)
print("pointer after  failed rollback :", maintain.read_pointer().name)
print("pointer moved:", maintain.read_pointer() != before)
```

Actual output:

```
BundleError: bundle was built with model_id 'foreign-embedder:v9', but the expected embedder is 'hash-bag-of-words:v1'; rebuild the index with the bundle's model before querying it
bundle built with embedder: C:\...\youtube_app\artifacts\v3-20260907T212149Z
pointer before rollback attempt: v3-20260907T212149Z
pointer after  failed rollback : v3-20260907T212149Z
pointer moved: False
```

(Refresh itself had moved the pointer to v3 because v3 validates against *its own*
embedder; the guard under test is that the *failed rollback* left the pointer where it
was. Note refresh validates against the refresh embedder — it cannot catch a
wrong-source transcript, only a bundle that cannot be reloaded. See Limitations.)

### 6. Restore the pointer and final state

```
$ PYTHONUTF8=1 python -m yt_rag.maintain --rollback v1
rolled back: current bundle is now C:\...\youtube_app\artifacts\v1-20260907T212022Z

$ python -m yt_rag.maintain --list
C:\...\youtube_app\artifacts\v1-20260907T212022Z  <- current
C:\...\youtube_app\artifacts\v2-20260907T212027Z
C:\...\youtube_app\artifacts\v3-20260907T212149Z
pointer: C:\...\youtube_app\artifacts\CURRENT
```

## Resolution and follow-up

- Pointer restored to v1; the bad v2 bundle was **not** deleted (bundles are never
  overwritten or rewritten in place — deleting old bundles is a manual cleanup decision).
- The v2 rebuild was repeatable: rebuilding from the correct fixture with a new label
  produces a new bundle and would be the forward fix; rollback restored service in one
  command in the meantime.
- Cover the guard in CI: `tests/test_maintain.py` asserts exactly the step-5 behavior
  (`test_rollback_identity_mismatch_leaves_pointer_untouched`).

## What this drill does NOT cover (honest scope)

- It is a simulation on committed fixtures, not a real outage; there is no production
  service and no on-call.
- The refresh path cannot detect that the *wrong source file* was ingested — identity
  validation covers embedder model_id/dim, not provenance. A low `top_score` after a
  refresh is the proxy that catches it (as it did here).
- All generation is `StubProvider`; no LLM answers were evaluated.
- Serving (`yt_rag.app`) is unchanged and still ingests per request; the pointer is a
  CLI-side artifact-management mechanism, not something the container currently reads.
- The real-embedder rebuild (pinned MiniLM) is the same command with `--real-embedder`,
  and downloads weights on first use (network) — not exercised in this drill.
