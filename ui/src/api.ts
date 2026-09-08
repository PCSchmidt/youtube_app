export interface RetrievedChunk {
  chunk_index: number;
  score: number;
  text: string;
}

export interface ChatResponse {
  question: string;
  answer: string;
  retrieved: RetrievedChunk[];
}

export interface AskResult extends ChatResponse {
  /** Client-observed round-trip time in ms (not server latency). */
  clientMs: number;
}

export interface ServerLatency {
  totalMeanMs: number | null;
  retrievalMeanMs: number | null;
  generationMeanMs: number | null;
  count: number;
}

export class ApiError extends Error {
  status: number;
  detail: string | null;
  constructor(status: number, message: string, detail: string | null = null) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

/** POST /chat and measure the client-side round trip. */
export async function askChat(
  source: { url?: string; file?: string },
  question: string,
): Promise<AskResult> {
  const started = performance.now();
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...source, question }),
  });
  const elapsed = Math.round(performance.now() - started);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      // keep the status-line detail
    }
    throw new ApiError(res.status, `The backend returned ${res.status}.`, detail);
  }
  const body = (await res.json()) as ChatResponse;
  return { ...body, clientMs: elapsed };
}

interface MetricsSnapshot {
  request_count?: number;
  latency_ms?: {
    total?: { mean_ms?: number };
    retrieval?: { mean_ms?: number };
    generation?: { mean_ms?: number };
  };
}

/** GET /metrics: server-side latency summary (since process start). */
export async function fetchServerLatency(): Promise<ServerLatency | null> {
  try {
    const res = await fetch("/metrics");
    if (!res.ok) return null;
    const body = (await res.json()) as MetricsSnapshot;
    return {
      totalMeanMs: body.latency_ms?.total?.mean_ms ?? null,
      retrievalMeanMs: body.latency_ms?.retrieval?.mean_ms ?? null,
      generationMeanMs: body.latency_ms?.generation?.mean_ms ?? null,
      count: body.request_count ?? 0,
    };
  } catch {
    return null;
  }
}

const MODES = new Set(["stub", "openai_compatible", "unknown"]);

/** GET /metrics/prometheus: parse the bounded provider-mode gauge. */
export async function fetchProviderMode(): Promise<string> {
  try {
    const res = await fetch("/metrics/prometheus");
    if (!res.ok) return "unknown (no response)";
    const text = await res.text();
    const m = text.match(/^yt_rag_provider_mode\{mode="([^"]+)"\} 1$/m);
    if (m && MODES.has(m[1])) return m[1];
    return "unknown (not exposed)";
  } catch {
    return "unknown (API not reachable)";
  }
}
