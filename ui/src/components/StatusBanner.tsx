interface Props {
  status: "idle" | "loading" | "success" | "error";
  message: string | null;
}

/** Single request-status region. Always rendered so it never shifts layout. */
export default function StatusBanner({ status, message }: Props) {
  return (
    <p
      className={`status-banner status-${status}`}
      role="status"
      aria-live="polite"
      data-testid="status-banner"
    >
      {status === "idle" && "Ask a question to see the pipeline run."}
      {status === "loading" && "Asking the RAG pipeline… (ingest → retrieve → generate)"}
      {status === "error" && `Request failed. ${message ?? ""}`}
      {status === "success" && "Answer ready."}
    </p>
  );
}
