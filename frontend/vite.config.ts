import { fileURLToPath, URL } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      // 将 /api 代理到后端，避免 CORS，且 SSE 流式可正常穿透
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
