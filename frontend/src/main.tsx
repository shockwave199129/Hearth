import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import { AlertProvider } from "./lib/alerts";
import { AlertStack } from "./components/AlertStack";
import { ProfileProvider } from "./lib/ProfileContext";
import { applyTheme, getStoredTheme } from "./lib/theme";
import { disableContextMenu } from "./lib/contextMenu";
import { blockDevtoolsShortcuts } from "./lib/devtools";
import "./styles/global.css";

// Applied once, synchronously, before any route renders — the persisted
// theme must survive landing on any page, not just Settings (where the
// toggle itself lives). See lib/theme.ts.
applyTheme(getStoredTheme());

// Registered here rather than in component effects so they're active before
// the first paint, and aren't torn down by a route change or StrictMode's
// double-mount. See lib/contextMenu.ts and lib/devtools.ts.
disableContextMenu();
blockDevtoolsShortcuts();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {/* Above the router so a toast fired right before navigate() survives
        the route change — see lib/alerts.tsx. */}
    <AlertProvider>
      <BrowserRouter>
        <ProfileProvider>
          <App />
        </ProfileProvider>
      </BrowserRouter>
      <AlertStack />
    </AlertProvider>
  </StrictMode>,
);
