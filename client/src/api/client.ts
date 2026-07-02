import type {
  DashboardStatus,
  HealthResponse,
  HistoryEvent,
  Task1AttemptSummary,
  Task1Bundle,
} from "../types/status";

export async function fetchHistory(limit = 100): Promise<HistoryEvent[]> {
  const res = await fetch(`/api/history?limit=${limit}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchTask1Attempts(): Promise<Task1AttemptSummary[]> {
  const res = await fetch("/api/task1/attempts");
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchTask1Attempt(id: string): Promise<Task1Bundle> {
  const res = await fetch(`/api/task1/attempts/${encodeURIComponent(id)}`);
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
