import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const proxyTarget = process.env.VITE_API_PROXY_TARGET ?? "http://localhost:8081";

export default defineConfig({
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 1200
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": proxyTarget,
      "/workspace": proxyTarget
    }
  }
});
