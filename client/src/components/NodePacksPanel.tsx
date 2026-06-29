import type { GpuSchedule } from "../types/status";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface NodePacksPanelProps {
  schedule: GpuSchedule;
}

export function NodePacksPanel({ schedule }: NodePacksPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>GPU node packing</CardTitle>
        <CardDescription>
          Pending jobs that could share one {schedule.gpus_per_node}-GPU node. Team view only.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {schedule.pending_packs.length === 0 ? (
          <p className="text-sm text-muted-foreground">Nothing queued to pack.</p>
        ) : (
          <ul className="grid gap-3 sm:grid-cols-2">
            {schedule.pending_packs.map((pack) => {
              const pct = Math.min(100, (pack.gpu_total / pack.gpus_per_node) * 100);
              return (
                <li key={pack.label} className="rounded-lg border border-border bg-muted/30 px-3 py-2.5">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
                      {pack.label}
                    </span>
                    <span className="font-mono text-xs text-muted-foreground tabular-nums">
                      {pack.gpu_total}/{pack.gpus_per_node} GPU
                    </span>
                  </div>
                  <p className="mt-1 font-mono text-sm text-foreground">{pack.job_ids.join("  +  ")}</p>
                  <Progress value={pct} className="mt-2 h-1.5" />
                </li>
              );
            })}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
