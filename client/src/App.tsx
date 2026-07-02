import { useState } from "react";
import type { DashboardStatus } from "./types/status";
import { ClusterPanel } from "./components/ClusterPanel";
import { CooldownPanel } from "./components/CooldownPanel";
import { FailedJobsPanel } from "./components/FailedJobsPanel";
import { Header } from "./components/Header";
import { HistoryPanel } from "./components/HistoryPanel";
import { Task1Viewer } from "./components/Task1Viewer";
import { LeaderboardPanel } from "./components/LeaderboardPanel";
import { NextActionsPanel } from "./components/NextActionsPanel";
import { NodePacksPanel } from "./components/NodePacksPanel";
import { OwnersPanel } from "./components/OwnersPanel";
import { ScoresPanel } from "./components/ScoresPanel";
import { SlurmQueue } from "./components/SlurmQueue";
import { StatStrip } from "./components/StatStrip";
import { useDashboard } from "./hooks/useDashboard";
import { useNotifications } from "./hooks/useNotifications";

const emptySchedule = {
  gpus_per_node: 4,
  running_gpus: 0,
  pending_gpus: 0,
  pending_packs: [],
};

function Banners({ status, error }: { status: DashboardStatus | null; error: string | null }) {
  const isMock = status?.mode === "mock";
  const warnings = status?.warnings ?? [];
  if (!isMock && !error && warnings.length === 0) return null;

  return (
    <div className="space-y-2">
      {error && (
        <p className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </p>
      )}
      {isMock && (
        <p className="rounded-lg border border-status-pending/30 bg-status-pending/10 px-3 py-2 text-xs text-status-pending">
          Mock data. On cluster, set{" "}
          <span className="font-mono">MODE = &quot;live&quot;</span> in{" "}
          <span className="font-mono">dashboard/config_local.py</span> (see{" "}
          <span className="font-mono">config_local.py.example</span>).
        </p>
      )}
      {warnings.map((w) => (
        <p
          key={w}
          className="rounded-lg border border-status-pending/30 bg-status-pending/5 px-3 py-2 text-xs text-status-pending"
        >
          {w}
        </p>
      ))}
    </div>
  );
}

export default function App() {
  const { data, error, loading, refresh } = useDashboard();
  const [notify, setNotify] = useState(false);
  useNotifications(data, notify);

  if (loading && !data) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="text-center">
          <span className="inline-block size-2 animate-pulse rounded-full bg-primary" />
          <p className="mt-3 font-mono text-xs uppercase tracking-[0.25em] text-muted-foreground">
            Pulling cluster state
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header
        status={data}
        onRefresh={refresh}
        notifyEnabled={notify}
        onToggleNotify={() => setNotify((v) => !v)}
      />

      <main className="mx-auto max-w-[1400px] px-6 py-6">
        <div className="space-y-6 duration-500 animate-in fade-in-0 slide-in-from-bottom-1">
          <Banners status={data} error={error} />
          <NextActionsPanel actions={data?.next_actions ?? []} />
          <FailedJobsPanel jobs={data?.failed_jobs ?? []} />
          <StatStrip
            summary={data?.gpu_summary ?? { used: 0, team_jobs: 0 }}
            schedule={data?.gpu_schedule ?? emptySchedule}
            cluster={data?.cluster ?? null}
          />

          <div className="grid gap-6 lg:grid-cols-3">
            <div className="space-y-6 lg:col-span-2">
              <SlurmQueue jobs={data?.slurm_jobs ?? []} />
              <NodePacksPanel schedule={data?.gpu_schedule ?? emptySchedule} />
            </div>
            <div className="space-y-6">
              <CooldownPanel tasks={data?.tasks ?? []} />
              <ScoresPanel tasks={data?.tasks ?? []} />
              <LeaderboardPanel rows={data?.leaderboard ?? []} />
            </div>
          </div>

          <HistoryPanel />
          <Task1Viewer />

          <div className="grid gap-6 md:grid-cols-2">
            <OwnersPanel owners={data?.owners ?? []} />
            <ClusterPanel cluster={data?.cluster ?? null} />
          </div>
        </div>
      </main>
    </div>
  );
}
