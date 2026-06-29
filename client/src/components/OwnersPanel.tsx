import type { OwnerSummary } from "../types/status";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface OwnersPanelProps {
  owners: OwnerSummary[];
}

export function OwnersPanel({ owners }: OwnersPanelProps) {
  if (owners.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>By teammate</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="divide-y divide-border">
          {owners.map((o) => (
            <li key={o.owner} className="py-3 text-sm first:pt-0 last:pb-0">
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-mono font-medium text-foreground">{o.owner}</span>
                <span className="font-mono text-xs text-muted-foreground tabular-nums">
                  {o.running_jobs}R · {o.pending_jobs}P
                </span>
              </div>
              {o.submit_ready_tasks.length > 0 && (
                <div className="mt-1 text-xs text-status-running">
                  Submit ready: {o.submit_ready_tasks.join(", ")}
                </div>
              )}
              {o.query_ready_tasks.length > 0 && (
                <div className="mt-0.5 text-xs text-primary">
                  Query ready: {o.query_ready_tasks.join(", ")}
                </div>
              )}
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
