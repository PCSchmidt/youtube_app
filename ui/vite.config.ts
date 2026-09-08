import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API port is configurable via API_PORT (default 8000, matching the FastAPI
// backend). Dev-server requests to /health, /metrics and /chat are proxied to
// it, so the UI works with no CORS setup.
const apiPort = process.env.API_PORT ?? "8000";
const apiTarget = `http://localhost:${apiPort}`;

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/health": apiTarget,
      "/metrics": apiTarget,
      "/chat": apiTarget,
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
  },
});
