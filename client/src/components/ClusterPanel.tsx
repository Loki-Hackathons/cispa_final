import type { ClusterStatus } from "../types/status";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface ClusterPanelProps {
  cluster: ClusterStatus | null;
}

function Cell({ label, value, accent }: { label: string; value: string | number; accent?: boolean }) {
  return (
    <div className="rounded-lg border border-border bg-muted/30 px-2 py-3 text-center">
      <p className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p>
      <p className={`mt-1 font-mono text-xl tabular-nums ${accent ? "text-status-running" : "text-foreground"}`}>
        {value}
      </p>
    </div>
  );
}

export function ClusterPanel({ cluster }: ClusterPanelProps) {
  if (!cluster) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Cluster</CardTitle>
        <CardDescription>Partition {cluster.partition}. Approximate node counts.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="grid grid-cols-3 gap-2">
          <Cell label="GPUs used" value={cluster.gpus_alloc ?? "—"} />
          <Cell label="GPUs idle" value={cluster.gpus_idle ?? "—"} accent />
          <Cell label="Team pending" value={cluster.team_pending} />
        </div>
        {cluster.note && <p className="text-xs text-muted-foreground">{cluster.note}</p>}
      </CardContent>
    </Card>
  );
}
