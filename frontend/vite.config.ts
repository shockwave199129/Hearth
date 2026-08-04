import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Host-side Vite defaults to loopback. In Docker Compose the backend is a
// sibling service — set VITE_BACKEND_PROXY=http://backend:48173.
const backendProxy = process.env.VITE_BACKEND_PROXY || "http://127.0.0.1:48173";
const backendWsProxy = backendProxy.replace(/^http/, "ws");

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 48176,
    proxy: {
      "/ws": { target: backendWsProxy, ws: true },
      "/api": { target: backendProxy },
    },
  },
});
