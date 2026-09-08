import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const SUCCESS_CHAT = {
  question: "how do I optimize my LinkedIn profile?",
  answer: "Rewrite your headline with recruiter keywords.",
  retrieved: [
    {
      chunk_index: 2,
      score: 0.41,
      text: "Rewrite your headline with the keywords recruiters search for.",
    },
  ],
};

function mockFetch(handler: (url: string, init?: RequestInit) => Promise<Response>) {
  const fn = vi.fn(async (url: string, init?: RequestInit) => handler(url, init));
  vi.stubGlobal("fetch", fn);
  return fn;
}

const jsonResponse = (body: unknown) =>
  new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
const textResponse = (body: string) => new Response(body, { status: 200 });

async function switchToLive(user: ReturnType<typeof userEvent.setup>) {
  await user.selectOptions(screen.getByLabelText(/Source/i), "live");
}

describe("yt_rag workspace UI", () => {
  it("shows the DEMO-labelled sample with no backend and no network", async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const user = userEvent.setup();
    render(<App />);
    expect(screen.getByRole("button", { name: /Run DEMO/i })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Run DEMO/i }));
    expect(await screen.findByRole("heading", { name: /Answer \(DEMO\)/i })).toBeInTheDocument();
    expect(screen.getAllByText(/DEMO/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Retrieved evidence \(4\)/i)).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("live mode: success state renders answer, evidence and evidence/answer distinction", async () => {
    const fetchSpy = mockFetch(async (url) => {
      if (url === "/chat") return jsonResponse(SUCCESS_CHAT);
      if (url === "/metrics") return jsonResponse({ request_count: 1, latency_ms: { total: { mean_ms: 5 } } });
      if (url === "/metrics/prometheus") return textResponse('yt_rag_provider_mode{mode="stub"} 1\n');
      throw new Error(`unexpected fetch ${url}`);
    });
    const user = userEvent.setup();
    render(<App />);
    await switchToLive(user);
    await user.type(screen.getByLabelText(/Question/i), "how do I optimize my LinkedIn profile?");
    await user.click(screen.getByRole("button", { name: /Ask the API/i }));
    expect(await screen.findByText(/Rewrite your headline with recruiter keywords\./)).toBeInTheDocument();
    expect(screen.getByText(/Retrieved evidence \(1\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Provider mode:/i)).toHaveTextContent("stub");
    expect(fetchSpy).toHaveBeenCalledWith("/chat", expect.objectContaining({ method: "POST" }));
  });

  it("live mode: loading state is announced and disables the button", async () => {
    let release: (v: Response) => void = () => {};
    mockFetch(
      () =>
        new Promise<Response>((resolve) => {
          release = resolve;
        }),
    );
    const user = userEvent.setup();
    render(<App />);
    await switchToLive(user);
    await user.type(screen.getByLabelText(/Question/i), "anything");
    await user.click(screen.getByRole("button", { name: /Ask the API/i }));
    expect(screen.getByTestId("status-banner")).toHaveTextContent(/Asking the RAG pipeline/i);
    expect(screen.getByRole("button", { name: /Working/i })).toBeDisabled();
    release(jsonResponse(SUCCESS_CHAT));
    await waitFor(() => expect(screen.getByTestId("status-banner")).toHaveTextContent(/Answer ready/i));
  });

  it("live mode: network error shows an actionable message", async () => {
    mockFetch(async () => {
      throw new TypeError("network down");
    });
    const user = userEvent.setup();
    render(<App />);
    await switchToLive(user);
    await user.type(screen.getByLabelText(/Question/i), "anything");
    await user.click(screen.getByRole("button", { name: /Ask the API/i }));
    expect(await screen.findByTestId("status-banner")).toHaveTextContent(/Could not reach the API/i);
    expect(screen.getByTestId("status-banner")).toHaveTextContent(/DEMO mode/i);
  });

  it("live mode: empty retrieved list shows the empty-result warning", async () => {
    mockFetch(async (url) => {
      if (url === "/chat") return jsonResponse({ ...SUCCESS_CHAT, retrieved: [] });
      if (url === "/metrics") return jsonResponse({});
      if (url === "/metrics/prometheus") return textResponse("");
      throw new Error(`unexpected fetch ${url}`);
    });
    const user = userEvent.setup();
    render(<App />);
    await switchToLive(user);
    await user.type(screen.getByLabelText(/Question/i), "obscure question");
    await user.click(screen.getByRole("button", { name: /Ask the API/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/No chunks were retrieved/i);
  });
});
