import { NavLink, Outlet, useLocation } from "react-router-dom";
import "./AppShell.css";
import { TierBadge } from "./TierBadge";
import { useTierStatus } from "../hooks/useTierStatus";
import { useProfile } from "../hooks/useProfile";
import { Chat } from "../pages/Chat";
import { Settings } from "../pages/Settings";

/** Shell keeps Chat mounted while Settings is open so the live transcript
 * and WebSocket survive Talk ↔ Settings navigation (Outlet alone would
 * unmount Chat and wipe in-memory turns). */
export function AppShell() {
  const { status, error: tierError } = useTierStatus();
  const { profile } = useProfile();
  const location = useLocation();
  const onChat = location.pathname === "/chat";
  const onSettings = location.pathname === "/settings";

  return (
    <div className="app-shell">
      <nav className="app-shell__rail" aria-label="Primary">
        <div className="app-shell__brand">{profile?.companion_name ?? "Companion"}</div>
        <div className="app-shell__links">
          <NavLink
            to="/chat"
            className={({ isActive }) => `app-shell__link${isActive ? " app-shell__link--active" : ""}`}
          >
            <OrbIcon />
            <span>Talk</span>
          </NavLink>
          <NavLink
            to="/settings"
            className={({ isActive }) => `app-shell__link${isActive ? " app-shell__link--active" : ""}`}
          >
            <GearIcon />
            <span>Settings</span>
          </NavLink>
        </div>
        <div className="app-shell__footer">
          <TierBadge status={status} error={tierError} />
        </div>
      </nav>
      <main className="app-shell__content">
        <div className="app-shell__view" hidden={!onChat}>
          <Chat />
        </div>
        <div className="app-shell__view" hidden={!onSettings}>
          <Settings />
        </div>
        {/* Keep RR outlet for nested routes if added later; chat/settings
            are rendered above so they stay mounted across nav. */}
        <Outlet />
      </main>
    </div>
  );
}

function OrbIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden>
      <circle cx="9" cy="9" r="6.5" stroke="currentColor" strokeWidth="1.4" />
      <circle cx="9" cy="9" r="2.5" fill="currentColor" />
    </svg>
  );
}

function GearIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden>
      <circle cx="9" cy="9" r="2.6" stroke="currentColor" strokeWidth="1.4" />
      <path
        d="M9 2.5v2M9 13.5v2M15.5 9h-2M4.5 9h-2M13.4 4.6l-1.4 1.4M6 10.6l-1.4 1.4M13.4 13.4l-1.4-1.4M6 7.4L4.6 6"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}
