import { useCallback, useState } from "react";
import {
  ApiError,
  askChat,
  fetchProviderMode,
  fetchServerLatency,
  type AskResult,
  type ServerLatency,
} from "./api";
import AnswerPanel from "./components/AnswerPanel";
import EvidencePanel from "./components/EvidencePanel";
import LatencyChart, { type LatencyPoint } from "./components/LatencyChart";
import ProviderModeBadge from "./components/ProviderModeBadge";
import StatusBanner from "./components/StatusBanner";
import { DEMO_SAMPLE } from "./demo/sample";

type Status = "idle" | "loading" | "success" | "error";
type SourceMode = "demo" | "live";

export default function App() {
  const [sourceMode, setSourceMode] = useState<SourceMode>("demo");
  const [url, setUrl] = useState("");
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<AskResult | null>(null);
  const [isDemoResult, setIsDemoResult] = useState(false);
  const [history, setHistory] = useState<LatencyPoint[]>([]);
  const [serverLatency, setServerLatency] = useState<ServerLatency | null>(null);
  const [providerMode, setProviderMode] = useState<string | null>(null);

  const runDemo = useCallback(() => {
    setIsDemoResult(true);
    setResult(DEMO_SAMPLE);
    setStatus("success");
    setErrorMessage(null);
    setServerLatency(null);
    setProviderMode("stub (demo)");
    setHistory((h) => [
      ...h,
      {
        label: `demo #${h.length + 1}`,
        clientMs: 0,
        retrievalMs: DEMO_SAMPLE.retrievalMs,
        generationMs: DEMO_SAMPLE.generationMs,
      },
    ]);
  }, []);

  const askLive = useCallback(async () => {
    const trimmed = question.trim();
    if (!trimmed) {
      setStatus("error");
      setErrorMessage("Enter a question first.");
      return;
    }
    setStatus("loading");
    setErrorMessage(null);
    const source = url.trim() ? { url: url.trim() } : {};
    try {
      const res = await askChat(source, trimmed);
      setResult(res);
      setIsDemoResult(false);
      setStatus("success");
      setHistory((h) => [
        ...h,
        { label: `q${h.length + 1}`, clientMs: res.clientMs, retrievalMs: null, generationMs: null },
      ]);
      // Server-side context is additive; failures here do not fail the ask.
      setServerLatency(await fetchServerLatency());
      setProviderMode(await fetchProviderMode());
    } catch (err) {
      setResult(null);
      setStatus("error");
      if (err instanceof ApiError) {
        setErrorMessage(
          `The backend returned ${err.status}: ${err.detail ?? err.message}. ${
            err.status === 422
              ? "Check the YouTube URL or use the DEMO mode."
              : "Check the backend logs for details."
          }`,
        );
      } else {
        setErrorMessage(
          "Could not reach the API at /chat. Is the FastAPI backend running (e.g. `docker compose up -d` or `uvicorn yt_rag.app:app`)? Use DEMO mode to explore the UI without a backend.",
        );
      }
    }
  }, [question, url]);

  const onSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (sourceMode === "demo") runDemo();
      else void askLive();
    },
    [sourceMode, runDemo, askLive],
  );

  return (
    <main className="workspace">
      <header className="workspace-header">
        <h1>yt_rag workspace</h1>
        <p className="tagline">
          Retrieval-augmented answers over YouTube transcripts. {isDemoResult ? "Showing DEMO data." : ""}
        </p>
        <ProviderModeBadge mode={providerMode} isDemo={isDemoResult} />
      </header>

      <div className="layout">
        <form className="panel ask-panel" onSubmit={onSubmit} aria-labelledby="ask-heading">
          <fieldset>
            <legend id="ask-heading">
              <h2>Ask</h2>
            </legend>

            <div className="field">
              <label htmlFor="source-mode">Source</label>
              <select
                id="source-mode"
                value={sourceMode}
                onChange={(e) => setSourceMode(e.target.value as SourceMode)}
              >
                <option value="demo">Bundled DEMO transcript (no backend)</option>
                <option value="live">Live API — YouTube URL</option>
              </select>
            </div>

            {sourceMode === "live" && (
              <div className="field">
                <label htmlFor="yt-url">YouTube URL or video ID</label>
                <input
                  id="yt-url"
                  type="text"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://www.youtube.com/watch?v=…"
                />
              </div>
            )}

            <div className="field">
              <label htmlFor="question">Question</label>
              <textarea
                id="question"
                rows={3}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g. How do I optimize my LinkedIn profile with ChatGPT?"
              />
            </div>

            <button type="submit" disabled={status === "loading"}>
              {status === "loading" ? "Working…" : sourceMode === "demo" ? "Run DEMO" : "Ask the API"}
            </button>

            <StatusBanner status={status} message={errorMessage} />
          </fieldset>
        </form>

        <div className="results" aria-busy={status === "loading"}>
          {result === null ? (
            <section className="panel empty-panel" aria-labelledby="empty-heading">
              <h2 id="empty-heading">No answer yet</h2>
              <p>
                Pick a source and ask a question. The DEMO mode shows a
                pre-rendered sample (labelled DEMO) without any backend; the
                live mode ingests a YouTube URL, retrieves top-k chunks, and
                generates from them.
              </p>
            </section>
          ) : (
            <>
              <AnswerPanel result={result} isDemo={isDemoResult} />
              <EvidencePanel retrieved={result.retrieved} />
            </>
          )}

          {serverLatency && (
            <section className="panel server-latency-panel" aria-labelledby="server-latency-heading">
              <h2 id="server-latency-heading">Server-side latency (since process start)</h2>
              <dl className="server-latency">
                <div>
                  <dt>Requests counted</dt>
                  <dd>{serverLatency.count}</dd>
                </div>
                <div>
                  <dt>Mean total</dt>
                  <dd>{serverLatency.totalMeanMs ?? "n/a"} ms</dd>
                </div>
                <div>
                  <dt>Mean retrieval</dt>
                  <dd>{serverLatency.retrievalMeanMs ?? "n/a"} ms</dd>
                </div>
                <div>
                  <dt>Mean generation</dt>
                  <dd>{serverLatency.generationMeanMs ?? "n/a"} ms</dd>
                </div>
              </dl>
            </section>
          )}

          <LatencyChart history={history} />
        </div>
      </div>
    </main>
  );
}
