import type { AskResult } from "../api";

export type SupportVerdict = "supported" | "partially supported" | "weakly supported";

/**
 * UI-side lexical heuristic ONLY. The API does not return a groundedness
 * field, so the UI compares answer wording against the retrieved evidence and
 * labels the result as a heuristic — it is NOT the backend's deterministic
 * groundedness evaluator and NOT semantic truth.
 */
export function assessSupport(answer: string, evidenceTexts: string[]): SupportVerdict {
  const stop = new Set([
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is",
    "are", "it", "that", "this", "you", "your", "be", "as", "at", "by", "from",
  ]);
  const words = (s: string) =>
    s
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 2 && !stop.has(w));
  const evidence = new Set(evidenceTexts.flatMap(words));
  const answerWords = words(answer);
  if (answerWords.length === 0) return "weakly supported";
  const hits = answerWords.filter((w) => evidence.has(w)).length;
  const ratio = hits / answerWords.length;
  if (ratio >= 0.5) return "supported";
  if (ratio >= 0.2) return "partially supported";
  return "weakly supported";
}

interface Props {
  result: AskResult;
  isDemo: boolean;
}

export default function AnswerPanel({ result, isDemo }: Props) {
  const verdict = assessSupport(
    result.answer,
    result.retrieved.map((c) => c.text),
  );
  const weak = verdict !== "supported";
  return (
    <section className="panel answer-panel" aria-labelledby="answer-heading">
      <h2 id="answer-heading">Answer {isDemo ? "(DEMO)" : ""}</h2>
      <p className={`support-verdict ${weak ? "support-weak" : "support-ok"}`}>
        <strong>Evidence support (UI-side heuristic):</strong> {verdict}.
        {weak && " The answer may not be grounded in the retrieved evidence."}
      </p>
      <blockquote className="answer-text">{result.answer}</blockquote>
    </section>
  );
}
