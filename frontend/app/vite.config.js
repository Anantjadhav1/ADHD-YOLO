import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The built dashboard is served by the FastAPI backend at /app/ (backend/app/main.py),
// so the page and the API share one origin and nothing has to be configured on the day.
export default defineConfig({
  base: "/app/",
  plugins: [react(), tailwindcss()],
});
