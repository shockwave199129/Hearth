import "./TierBadge.css";
import type { TierStatus } from "../hooks/useTierStatus";

interface TierBadgeProps {
  status: TierStatus | null;
  error?: string | null;
}

const TIER_COPY: Record<string, string> = {
  S: "Full quality — dedicated GPU",
  A: "High quality",
  B: "Balanced for this machine",
  C: "Lightweight mode",
};

export function TierBadge({ status, error }: TierBadgeProps) {
  if (error) {
    return <div className="tier-badge tier-badge--pending">Hardware check failed</div>;
  }
  if (!status) {
    return <div className="tier-badge tier-badge--pending">Detecting hardware…</div>;
  }
  return (
    <div className={`tier-badge tier-badge--${status.tier.toLowerCase()}`}>
      <span className="tier-badge__dot" aria-hidden />
      <span className="tier-badge__tier">Tier {status.tier}</span>
      <span className="tier-badge__desc">{TIER_COPY[status.tier] ?? "Configured"}</span>
    </div>
  );
}
