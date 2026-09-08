import type { AskResult } from "../api";

export interface DemoResult extends AskResult {
  retrievalMs: number;
  generationMs: number;
}

/**
 * Bundled, pre-rendered sample so the UI is demonstrable with no backend and
 * no network. Always presented with a visible DEMO label; these numbers are
 * fixed fixture data, not live measurements.
 */
export const DEMO_SAMPLE: DemoResult = {
  question: "How do I optimize my LinkedIn profile with ChatGPT?",
  answer:
    "[stub answer, grounded in 3 retrieved chunk(s)]\nRelevant transcript passage (chunk 2): The fastest way to improve your LinkedIn profile is to rewrite your headline with the keywords recruiters search for, then use ChatGPT to draft a short About section that states what you do, who you help, and the outcome you deliver.",
  retrieved: [
    {
      chunk_index: 2,
      score: 0.4132,
      text:
        "The fastest way to improve your LinkedIn profile is to rewrite your headline with the keywords recruiters actually search for. Keep it specific: role, domain, outcome.",
    },
    {
      chunk_index: 5,
      score: 0.3421,
      text:
        "Use ChatGPT to draft a short About section that states what you do, who you help, and the outcome you deliver. Ask it for three variants and pick the clearest one.",
    },
    {
      chunk_index: 7,
      score: 0.2874,
      text:
        "LinkedIn rewards complete profiles: a photo, a custom URL, and three to five skills endorsed by real people. Small hygiene fixes beat any clever prompt.",
    },
    {
      chunk_index: 1,
      score: 0.2109,
      text:
        "This transcript walks through optimizing a LinkedIn profile step by step, starting with the headline and the About section.",
    },
  ],
  clientMs: 0,
  retrievalMs: 0.13,
  generationMs: 0.01,
};
