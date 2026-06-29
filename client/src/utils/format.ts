import type { SlurmJob } from "../types/status";

export function formatDuration(seconds: number | null): string {
  if (seconds === null || seconds < 0) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function formatCooldown(seconds: number): string {
  if (seconds <= 0) return "ready";
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

export function formatEtaLabel(kind: SlurmJob["eta_kind"]): string {
  if (kind === "time_limit_remaining") return "Limit left";
  if (kind === "scheduled_start") return "Est. start";
  if (kind === "reported") return "Reported";
  if (kind === "extrapolated") return "Extrapolated";
  if (kind === "stale_progress") return "Stalled?";
  return "ETA";
}

export function formatScoreDelta(delta: number | null): string {
  if (delta === null) return "";
  const sign = delta >= 0 ? "+" : "";
  return `${sign}${delta.toFixed(4)}`;
}
