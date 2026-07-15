import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";
import * as notifications from "./notifications";

export type AlertType = "success" | "error";

export interface Alert {
  id: string;
  type: AlertType;
  message: string;
}

interface AlertContextValue {
  alerts: Alert[];
  showAlert: (alert: { type: AlertType; message: string }) => void;
  dismissAlert: (id: string) => void;
}

const AlertContext = createContext<AlertContextValue | null>(null);

const AUTO_DISMISS_MS = 4000;

/** Wraps the whole app (see main.tsx) — above the router, so a toast fired
 * right before a navigate() call is still visible after the route change. */
export function AlertProvider({ children }: { children: ReactNode }) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const timers = useRef(new Map<string, ReturnType<typeof setTimeout>>());

  const dismissAlert = useCallback((id: string) => {
    setAlerts((prev) => prev.filter((a) => a.id !== id));
    const timer = timers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
  }, []);

  const showAlert = useCallback(
    ({ type, message }: { type: AlertType; message: string }) => {
      const id = crypto.randomUUID();
      setAlerts((prev) => [...prev, { id, type, message }]);
      timers.current.set(
        id,
        setTimeout(() => dismissAlert(id), AUTO_DISMISS_MS),
      );

      // Only adds an OS notification when the user isn't looking at the
      // app at all — otherwise the toast above is already enough, and a
      // simultaneous OS popup would just be a redundant second alert for
      // the same event. Never requests permission itself — see the
      // Settings "Desktop notifications" toggle for the consent step.
      notifications.notify(document.title, message);
    },
    [dismissAlert],
  );

  return (
    <AlertContext.Provider value={{ alerts, showAlert, dismissAlert }}>{children}</AlertContext.Provider>
  );
}

export function useAlert(): AlertContextValue {
  const ctx = useContext(AlertContext);
  if (!ctx) throw new Error("useAlert must be used within an AlertProvider");
  return ctx;
}
