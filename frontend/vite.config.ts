import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 48176,
    proxy: {
      "/ws": { target: "ws://127.0.0.1:48173", ws: true },
      "/api": { target: "http://127.0.0.1:48173" },
    },
  },
});
