import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { Chat } from "./pages/Chat";
import { Onboarding } from "./pages/Onboarding";
import { Setup } from "./pages/Setup";
import { Settings } from "./pages/Settings";
import { useProfile } from "./hooks/useProfile";
import { useSetupStatus } from "./hooks/useSetupStatus";
import "./App.css";

export function App() {
  const setup = useSetupStatus();
  // Runs unconditionally alongside the setup gate below (React hooks can't
  // be called conditionally) — /api/profile doesn't need Pipeline() to be
  // built, so this resolves fine even before setup completes; its result
  // just isn't used for anything until we fall through past the gate.
  const { profile, loading, error } = useProfile();

  if (setup.status === null && setup.error === null) {
    return (
      <div className="app-splash" aria-busy="true">
        <p className="app-splash__message">Checking your setup…</p>
      </div>
    );
  }

  if (setup.error) {
    return (
      <div className="app-splash">
        <p className="app-splash__message">{setup.error}</p>
      </div>
    );
  }

  if (setup.status && !setup.status.complete) {
    return (
      <Setup
        status={setup.status}
        statusError={setup.error}
        progress={setup.progress}
        starting={setup.starting}
        onStart={() => void setup.startSetup()}
      />
    );
  }

  if (loading) {
    // Stays true throughout useProfile's retry-with-backoff window (see
    // lib/backendFetch.ts) — a packaged app's backend can take a while to
    // come up (local LLM load, etc.), so this covers that whole startup
    // period rather than flashing an error the moment the first request
    // loses the race.
    return (
      <div className="app-splash" aria-busy="true">
        <p className="app-splash__message">Waking up your companion…</p>
      </div>
    );
  }

  // A backend the app still can't reach after those retries are exhausted
  // shouldn't force the user through onboarding again — fall through to the
  // normal routes and let Chat/Settings surface the connection problem instead.
  const onboarded = profile !== null || error !== null;

  return (
    <Routes>
      <Route path="/" element={<Navigate to={onboarded ? "/chat" : "/onboarding"} replace />} />
      <Route path="/onboarding" element={<Onboarding />} />
      <Route element={<AppShell />}>
        <Route path="/chat" element={<Chat />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
