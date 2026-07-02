import type { DashboardStatus, HealthResponse, HistoryEvent } from "../types/status";

export async function fetchHistory(limit = 100): Promise<HistoryEvent[]> {
  const res = await fetch(`/api/history?limit=${limit}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchStatus(): Promise<DashboardStatus> {
  const res = await fetch("/api/status");
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}
