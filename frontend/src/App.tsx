import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { Chat } from "./pages/Chat";
import { Onboarding } from "./pages/Onboarding";
import { Settings } from "./pages/Settings";
import { useProfile } from "./hooks/useProfile";
import "./App.css";

export function App() {
  const { profile, loading, error } = useProfile();

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
