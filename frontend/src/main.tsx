import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import { AlertProvider } from "./lib/alerts";
import { AlertStack } from "./components/AlertStack";
import { applyTheme, getStoredTheme } from "./lib/theme";
import "./styles/global.css";

// Applied once, synchronously, before any route renders — the persisted
// theme must survive landing on any page, not just Settings (where the
// toggle itself lives). See lib/theme.ts.
applyTheme(getStoredTheme());

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {/* Above the router so a toast fired right before navigate() survives
        the route change — see lib/alerts.tsx. */}
    <AlertProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
      <AlertStack />
    </AlertProvider>
  </StrictMode>,
);
