import type { TaskStatus } from "../types/status";
import { formatCooldown } from "../utils/format";
import { CommandChips } from "./CommandChips";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface CooldownPanelProps {
  tasks: TaskStatus[];
}

function GateRow({
  label,
  ready,
  seconds,
  maxSeconds,
}: {
  label: string;
  ready: boolean;
  seconds: number;
  maxSeconds: number;
}) {
  const pct = ready ? 100 : Math.max(0, 100 - (seconds / maxSeconds) * 100);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="font-mono uppercase tracking-wider text-muted-foreground">{label}</span>
        <span
          className={cn(
            "font-mono tabular-nums",
            ready ? "font-medium text-status-running" : "text-status-pending",
          )}
        >
          {ready ? "open" : formatCooldown(seconds)}
        </span>
      </div>
      <Progress
        value={pct}
        className={cn(
          "h-1",
          ready
            ? "[&_[data-slot=progress-indicator]]:bg-status-running"
            : "[&_[data-slot=progress-indicator]]:bg-status-pending",
        )}
      />
    </div>
  );
}

export function CooldownPanel({ tasks }: CooldownPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>API gates</CardTitle>
        <CardDescription>Submit and query cooldowns per task.</CardDescription>
      </CardHeader>
      <CardContent>
        {tasks.length === 0 ? (
          <p className="text-sm text-muted-foreground">No task state yet.</p>
        ) : (
          <ul className="divide-y divide-border">
            {tasks.map((task) => (
              <li key={task.task_id} className="py-3 first:pt-0 last:pb-0">
                <div className="mb-3 flex items-baseline justify-between gap-2">
                  <span className="font-mono text-sm font-medium text-foreground">{task.task_id}</span>
                  <span className="font-mono text-[10px] text-muted-foreground">
                    {task.owner ?? "—"}
                  </span>
                </div>
                <div className="space-y-2.5">
                  <GateRow
                    label="submit"
                    ready={task.submit_ready}
                    seconds={task.submit_cooldown_seconds}
                    maxSeconds={300}
                  />
                  <GateRow
                    label="query"
                    ready={task.query_ready}
                    seconds={task.query_cooldown_seconds}
                    maxSeconds={900}
                  />
                </div>
                <CommandChips commands={task.commands} />
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
