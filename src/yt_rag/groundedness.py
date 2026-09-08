"""Deterministic, offline groundedness evaluation for generated answers.

Claim-level lexical evaluator. No LLM judge, no network, no model download:
given an answer and the retrieved context texts it was (supposedly) grounded
on, it classifies the answer as ``empty``, ``evasive``, ``supported``,
``partially_supported``, or ``unsupported``.

How it works (fully deterministic heuristics):

1. Extract claims: the answer is split into sentences. Known StubProvider
   wrapper lines ("[stub answer, ...]" and "Relevant transcript passage
   (chunk N):") are stripped as metadata, not scored.
2. Classify each claim: a claim is *supported* when at least
   ``SUPPORT_THRESHOLD`` of its content words appear in the context, AND
   every number in the claim also appears in the context. Otherwise it is
   *unsupported*. Missing numbers are the only available
   fabricated-figure/contradiction signal.
3. Classify the answer: empty (no text), evasive (a refusal phrase, e.g.
   "cannot find it in the transcript"), then by claim verdicts.

LIMITATIONS (these are heuristic labels, NOT semantic truth):

- Lexical overlap is not meaning. A claim can reuse context words and still
  be false (negation, reversed relations, swapped entities), and a true
  paraphrase with different words can score unsupported.
- Numbers are compared as strings; "220" and "two hundred twenty" do not
  match, and "220" in the context does not mean the claim uses it correctly.
- There is no semantic contradiction detection: "X is 220 characters" is
  supported by context mentioning 220 even if the truth is 500. Only numbers
  (or words) *absent* from the context are flagged.
- Relevance to the question is not measured at all.
- "supported" means "built from retrieved text", not "true".
- The metadata stripper is tuned to the known StubProvider wrapper; other
  generators' boilerplate would be scored as claims.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from yt_rag.config import REPO_ROOT

DEFAULT_GROUNDEDNESS_SET = REPO_ROOT / "experiments" / "groundedness_set.json"

# A claim is supported when at least this fraction of its content words appear
# in the retrieved context (and all its numbers appear verbatim).
SUPPORT_THRESHOLD = 0.6

# Phrases that indicate the generator said the answer is not in the transcript.
EVASIVE_PHRASES = (
    "cannot find it in the transcript",
    "can't find it in the transcript",
    "not in the transcript",
    "not mentioned in the transcript",
    "does not mention",
    "do not mention",
    "no information about",
    "i don't know",
    "i do not know",
)

# Known generator wrapper lines that carry no claim content (StubProvider).
_METADATA_LINE_RE = re.compile(r"^\s*(\[stub answer|relevant transcript passage)", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# Content words scored by the evaluator. Answers containing ONLY these words
# carry no checkable claim.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)

# Honest one-line summary of what the numbers do and do not mean, embedded in
# every run record so the limitation travels with the data.
GROUNDEDNESS_LIMITATIONS = (
    "lexical/heuristic claim-overlap labels, NOT semantic truth: coverage of "
    "context words does not prove a claim is correct, paraphrases can score "
    "unsupported, and only numbers/words ABSENT from the context are flagged "
    "(no contradiction detection); question relevance is not measured"
)


@dataclass(frozen=True)
class ClaimVerdict:
    """One extracted claim and its lexical support verdict."""

    text: str
    verdict: str  # "supported" | "unsupported"
    unsupported_tokens: tuple[str, ...]
    coverage: float  # fraction of content tokens found in the context


@dataclass(frozen=True)
class GroundednessResult:
    """Answer-level groundedness verdict plus per-claim detail."""

    verdict: str  # empty | evasive | supported | partially_supported | unsupported
    evasion_detected: bool
    n_claims: int
    n_supported: int
    n_unsupported: int
    claim_groundedness: float | None  # None when there are no claims
    claims: tuple[ClaimVerdict, ...]

    def to_dict(self) -> dict:
        """JSON-serializable form for run records (claims stay compact)."""
        return {
            "verdict": self.verdict,
            "evasion_detected": self.evasion_detected,
            "n_claims": self.n_claims,
            "n_supported": self.n_supported,
            "n_unsupported": self.n_unsupported,
            "claim_groundedness": self.claim_groundedness,
            "unsupported_tokens": sorted({t for c in self.claims for t in c.unsupported_tokens}),
        }


def content_tokens(text: str) -> list[str]:
    """Lowercased content words (no stopwords, punctuation split off)."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def extract_claims(answer: str) -> list[str]:
    """Split an answer into non-empty claim sentences.

    Known StubProvider wrapper lines are dropped as metadata. Claims with no
    content tokens (pure stopword/filler sentences) are dropped too.
    """
    lines = [
        line for line in answer.splitlines() if line.strip() and not _METADATA_LINE_RE.match(line)
    ]
    text = " ".join(lines)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip() and content_tokens(p)]


def classify_claim(claim: str, context_lower: str) -> ClaimVerdict:
    """Lexically classify one claim against pre-lowercased context text.

    Supported = coverage >= SUPPORT_THRESHOLD and every number in the claim
    appears verbatim in the context (the fabricated-figure signal).
    """
    tokens = content_tokens(claim)
    missing = [t for t in tokens if t not in context_lower]
    coverage = 1.0 - len(missing) / len(tokens)
    # Numbers are content tokens too, so a missing number is already in
    # `missing`; the explicit check just enforces it regardless of coverage.
    numbers = [t for t in tokens if _NUMBER_RE.fullmatch(t)]
    missing_numbers = [n for n in numbers if n not in context_lower]
    supported = coverage >= SUPPORT_THRESHOLD and not missing_numbers
    return ClaimVerdict(
        text=claim,
        verdict="supported" if supported else "unsupported",
        unsupported_tokens=tuple(dict.fromkeys(missing)),
        coverage=coverage,
    )


def evasion_detected(answer: str) -> bool:
    """True if the answer is a cannot-find/refusal style response."""
    lowered = answer.lower()
    return any(phrase in lowered for phrase in EVASIVE_PHRASES)


def evaluate_groundedness(answer: str, context_texts: list[str]) -> GroundednessResult:
    """Classify an answer against the retrieved context it was grounded on.

    Precedence: empty > evasive > claim verdicts. An evasive answer is not
    scored for claims even if it also contains some (documented limitation:
    refuse-then-answer behavior is treated as refusal).
    """
    answer = answer or ""
    if not answer.strip():
        return GroundednessResult(
            verdict="empty",
            evasion_detected=False,
            n_claims=0,
            n_supported=0,
            n_unsupported=0,
            claim_groundedness=None,
            claims=(),
        )
    evasive = evasion_detected(answer)
    if evasive:
        return GroundednessResult(
            verdict="evasive",
            evasion_detected=True,
            n_claims=0,
            n_supported=0,
            n_unsupported=0,
            claim_groundedness=None,
            claims=(),
        )
    context_lower = " ".join(context_texts).lower()
    claim_verdicts = tuple(classify_claim(c, context_lower) for c in extract_claims(answer))
    n_supported = sum(1 for c in claim_verdicts if c.verdict == "supported")
    n_unsupported = len(claim_verdicts) - n_supported
    if not claim_verdicts:
        verdict = "unsupported"  # non-empty, non-evasive, but nothing checkable
        groundedness: float | None = None
    elif n_unsupported == 0:
        verdict = "supported"
        groundedness = 1.0
    elif n_supported == 0:
        verdict = "unsupported"
        groundedness = 0.0
    else:
        verdict = "partially_supported"
        groundedness = n_supported / len(claim_verdicts)
    return GroundednessResult(
        verdict=verdict,
        evasion_detected=False,
        n_claims=len(claim_verdicts),
        n_supported=n_supported,
        n_unsupported=n_unsupported,
        claim_groundedness=groundedness,
        claims=claim_verdicts,
    )


# --- labeled groundedness set ---------------------------------------------------


def load_groundedness_set(path: str | Path | None = None) -> dict:
    """Load the committed, hand-labeled groundedness cases."""
    p = Path(path) if path else DEFAULT_GROUNDEDNESS_SET
    return json.loads(p.read_text(encoding="utf-8"))


def run_groundedness_set(path: str | Path | None = None) -> dict:
    """Run the evaluator over the labeled set and report agreement.

    Every case carries the expected answer-level verdict, labeled by the
    author against the deterministic evaluator's documented behavior. This
    checks the evaluator, NOT answer truth (see module limitations).
    """
    data = load_groundedness_set(path)
    per_case = []
    n_agree = 0
    for case in data["cases"]:
        result = evaluate_groundedness(case["answer"], case["context"])
        ok = result.verdict == case["expected_verdict"]
        n_agree += ok
        per_case.append(
            {
                "id": case["id"],
                "category": case["category"],
                "expected_verdict": case["expected_verdict"],
                "verdict": result.verdict,
                "n_claims": result.n_claims,
                "n_supported": result.n_supported,
                "n_unsupported": result.n_unsupported,
                "ok": ok,
            }
        )
    n = len(per_case)
    return {
        "path": str(Path(path) if path else DEFAULT_GROUNDEDNESS_SET),
        "n_cases": n,
        "n_agree": n_agree,
        "agreement": (n_agree / n) if n else None,
        "limitations": GROUNDEDNESS_LIMITATIONS,
        "per_case": per_case,
    }
