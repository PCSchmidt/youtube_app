import type { RetrievedChunk } from "../api";

interface Props {
  retrieved: RetrievedChunk[];
}

/** Retrieved chunks: distinct from the generated answer by design. */
export default function EvidencePanel({ retrieved }: Props) {
  return (
    <section className="panel evidence-panel" aria-labelledby="evidence-heading">
      <h2 id="evidence-heading">Retrieved evidence ({retrieved.length})</h2>
      {retrieved.length === 0 ? (
        <p className="empty-result-warning" role="alert">
          No chunks were retrieved for this question, so the answer is not
          grounded in any transcript text. Try rephrasing or checking the
          source.
        </p>
      ) : (
        <ol className="evidence-list">
          {retrieved.map((chunk) => (
            <li key={chunk.chunk_index} className="evidence-item">
              <p className="evidence-meta">
                chunk #{chunk.chunk_index} — cosine similarity{" "}
                <meter min={0} max={1} value={chunk.score} title="cosine similarity" />{" "}
                {chunk.score.toFixed(3)}
              </p>
              <p className="evidence-text">{chunk.text}</p>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
