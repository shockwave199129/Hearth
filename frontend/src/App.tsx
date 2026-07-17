import { Navigate, Route, Routes, useSearchParams } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { Onboarding } from "./pages/Onboarding";
import { Setup } from "./pages/Setup";
import { useProfile } from "./hooks/useProfile";
import { useSetupStatus } from "./hooks/useSetupStatus";
import "./App.css";

/** First-run onboarding when no profile; `?mode=add` allows creating another
 * profile from Settings even when one already exists. */
function OnboardingRoute() {
  const { profile } = useProfile();
  const [params] = useSearchParams();
  const addingAnother = params.get("mode") === "add";

  if (profile !== null && !addingAnother) {
    return <Navigate to="/chat" replace />;
  }
  return <Onboarding />;
}

export function App() {
  const setup = useSetupStatus();
  // Runs unconditionally alongside the setup gate below (React hooks can't
  // be called conditionally) — /api/profile doesn't need Pipeline() to be
  // built, so this resolves fine even before setup completes; its result
  // just isn't used for anything until we fall through past the gate.
  // Shared via ProfileProvider so Onboarding's submit updates this same state.
  const { profile, loading, error } = useProfile();

  if (setup.status === null && setup.error === null) {
    return (
      <div className="app-splash" aria-busy="true">
        <p className="app-splash__message">Checking your setup…</p>
      </div>
    );
  }

  // Connection / spawn failures are retryable — show Setup with Retry
  // instead of a dead-end splash. Incomplete setup is the normal first-run path.
  if (setup.error || (setup.status && !setup.status.complete)) {
    return (
      <Setup
        status={setup.status}
        statusError={setup.error}
        progress={setup.progress}
        starting={setup.starting}
        onStart={() => void setup.startSetup()}
        onRetryStatus={() => setup.retryStatus()}
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

  // Active profile in profile.db is the source of truth — once onboarding
  // has saved one, later launches go straight to chat. A backend the app
  // still can't reach after retries shouldn't force the onboarding wizard
  // again either; Chat/Settings surface the connection problem instead.
  const hasProfile = profile !== null;
  const skipOnboarding = hasProfile || error !== null;

  return (
    <Routes>
      <Route path="/" element={<Navigate to={skipOnboarding ? "/chat" : "/onboarding"} replace />} />
      <Route path="/onboarding" element={<OnboardingRoute />} />
      <Route element={<AppShell />}>
        <Route path="/chat" element={<></>} />
        <Route path="/settings" element={<></>} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
