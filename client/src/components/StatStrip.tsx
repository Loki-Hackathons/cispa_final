import type { ClusterStatus, GpuSchedule, GpuSummary } from "../types/status";
import { cn } from "@/lib/utils";

interface StatStripProps {
  summary: GpuSummary;
  schedule: GpuSchedule;
  cluster: ClusterStatus | null;
}

interface Tile {
  label: string;
  value: number | null;
  accent?: "running" | "pending";
}

function StatTile({ label, value, accent }: Tile) {
  return (
    <div className="rounded-xl border border-border bg-card px-4 py-3">
      <p className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">{label}</p>
      <p
        className={cn(
          "mt-1.5 font-mono text-2xl font-medium leading-none tabular-nums",
          accent === "running" && (value ?? 0) > 0 && "text-status-running",
          accent === "pending" && (value ?? 0) > 0 && "text-status-pending",
        )}
      >
        {value ?? "—"}
      </p>
    </div>
  );
}

export function StatStrip({ summary, schedule, cluster }: StatStripProps) {
  const tiles: Tile[] = [
    { label: "GPUs live", value: summary.used },
    { label: "Active jobs", value: summary.team_jobs },
    { label: "Pending GPUs", value: schedule.pending_gpus, accent: "pending" },
    { label: "Cluster idle", value: cluster?.gpus_idle ?? null, accent: "running" },
    { label: "Per node", value: schedule.gpus_per_node },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      {tiles.map((tile) => (
        <StatTile key={tile.label} {...tile} />
      ))}
    </div>
  );
}
