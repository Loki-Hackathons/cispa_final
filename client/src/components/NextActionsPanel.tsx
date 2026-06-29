import type { NextAction } from "../types/status";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface NextActionsPanelProps {
  actions: NextAction[];
}

const dotColor: Record<string, string> = {
  job_failed: "bg-destructive",
  submit_ready: "bg-status-running",
  query_ready: "bg-primary",
  cooldown_soon: "bg-status-pending",
  job_stalled: "bg-status-pending",
  node_pack: "bg-muted-foreground",
};

export function NextActionsPanel({ actions }: NextActionsPanelProps) {
  if (actions.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Next actions</CardTitle>
        <CardDescription>Suggested moves from cooldowns, queue, and failures.</CardDescription>
      </CardHeader>
      <CardContent>
        <ul className="grid gap-2 sm:grid-cols-2">
          {actions.map((action, i) => (
            <li
              key={`${action.kind}-${i}`}
              className="flex items-start gap-2.5 rounded-lg border border-border bg-muted/20 px-3 py-2 text-sm"
            >
              <span
                className={`mt-1.5 size-2 shrink-0 rounded-full ${dotColor[action.kind] ?? "bg-muted-foreground"}`}
                aria-hidden
              />
              <span className="text-foreground">{action.message}</span>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
