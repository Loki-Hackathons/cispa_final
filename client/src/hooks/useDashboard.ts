import { useCallback, useEffect, useState } from "react";
import { fetchStatus } from "../api/client";
import type { DashboardStatus } from "../types/status";

interface UseDashboardResult {
  data: DashboardStatus | null;
  error: string | null;
  loading: boolean;
  refresh: () => void;
}

export function useDashboard(): UseDashboardResult {
  const [data, setData] = useState<DashboardStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const status = await fetchStatus();
      setData(status);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch status");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    const interval = (data?.refresh_seconds ?? 5) * 1000;
    const id = setInterval(refresh, interval);
    return () => clearInterval(id);
  }, [data?.refresh_seconds, refresh]);

  return { data, error, loading, refresh };
}
