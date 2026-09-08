import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface LatencyPoint {
  label: string;
  clientMs: number;
  retrievalMs: number | null;
  generationMs: number | null;
}

interface Props {
  history: LatencyPoint[];
}

/** Client-observed round trip per question (ms), charted with labeled axes. */
export default function LatencyChart({ history }: Props) {
  if (history.length === 0) return null;
  const summary = history
    .map((p) => `${p.label}: ${p.clientMs} ms round trip`)
    .join("; ");
  const showDemoSeries = history.some(
    (p) => p.retrievalMs !== null || p.generationMs !== null,
  );
  return (
    <section className="panel latency-panel" aria-labelledby="latency-heading">
      <h2 id="latency-heading">Request latency</h2>
      <figure>
        <div
          role="img"
          aria-label={`Bar chart of client-observed round-trip latency in milliseconds. ${summary}`}
          style={{ width: "100%", height: 220 }}
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={history} margin={{ top: 8, right: 8, bottom: 24, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" />
              <YAxis width={56} label={{ value: "ms", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="clientMs" name="Client round trip (ms)" fill="#3b6ea5" />
              {showDemoSeries && (
                <>
                  <Bar dataKey="retrievalMs" name="Retrieval (ms, demo)" fill="#7fa8d1" />
                  <Bar dataKey="generationMs" name="Generation (ms, demo)" fill="#b5cde3" />
                </>
              )}
            </BarChart>
          </ResponsiveContainer>
        </div>
        <figcaption>
          Units are milliseconds. Client round trip is measured in the browser;
          the retrieval/generation series appear only for DEMO values. Live
          server-side latency summaries (since process start) are listed under
          the chart when the API is reachable.
        </figcaption>
      </figure>
    </section>
  );
}
