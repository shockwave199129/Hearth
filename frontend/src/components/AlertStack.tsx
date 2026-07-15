import "./AlertStack.css";
import { useAlert } from "../lib/alerts";

/** Rendered once at app root, alongside AlertProvider — see main.tsx. */
export function AlertStack() {
  const { alerts, dismissAlert } = useAlert();

  if (alerts.length === 0) return null;

  return (
    <div className="alert-stack" role="status" aria-live="polite">
      {alerts.map((alert) => (
        <div key={alert.id} className={`alert-stack__item alert-stack__item--${alert.type}`}>
          <span className="alert-stack__message">{alert.message}</span>
          <button
            type="button"
            className="alert-stack__dismiss"
            onClick={() => dismissAlert(alert.id)}
            aria-label="Dismiss"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
