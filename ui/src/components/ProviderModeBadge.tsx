interface Props {
  mode: string | null;
  isDemo: boolean;
}

/**
 * Live value comes from GET /metrics/prometheus (yt_rag_provider_mode gauge).
 * In DEMO mode there is no backend, so the badge says so honestly.
 */
export default function ProviderModeBadge({ mode, isDemo }: Props) {
  if (isDemo) {
    return (
      <p className="provider-badge provider-demo">
        Provider mode: <strong>stub (DEMO sample, no backend)</strong>
      </p>
    );
  }
  return (
    <p className="provider-badge">
      Provider mode: <strong>{mode ?? "checking…"}</strong>
      {mode === "unknown (API not reachable)" && " — is the backend running?"}
    </p>
  );
}
