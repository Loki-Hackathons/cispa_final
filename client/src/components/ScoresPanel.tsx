import type { TaskStatus } from "../types/status";
import { formatScoreDelta } from "../utils/format";
import { CommandChips } from "./CommandChips";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

interface ScoresPanelProps {
  tasks: TaskStatus[];
}

const HEAD = "font-mono text-[10px] uppercase tracking-wider text-muted-foreground";

export function ScoresPanel({ tasks }: ScoresPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Scores</CardTitle>
      </CardHeader>
      <CardContent className="px-0">
        {tasks.length === 0 ? (
          <p className="px-4 text-sm text-muted-foreground">No submissions yet.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className={cn(HEAD, "pl-4")}>Task</TableHead>
                <TableHead className={HEAD}>Score</TableHead>
                <TableHead className={HEAD}>Δ</TableHead>
                <TableHead className={HEAD}>Try</TableHead>
                <TableHead className={cn(HEAD, "pr-4")}>W&amp;B</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {tasks.map((task) => (
                <TableRow key={task.task_id} className="align-top">
                  <TableCell className="pl-4 font-mono text-sm">{task.task_id}</TableCell>
                  <TableCell className="font-mono text-sm font-medium tabular-nums">
                    {task.last_score !== null ? task.last_score.toFixed(6) : "—"}
                  </TableCell>
                  <TableCell
                    className={cn(
                      "font-mono text-sm tabular-nums",
                      task.score_delta != null && task.score_delta >= 0 && "text-status-running",
                      task.score_delta != null && task.score_delta < 0 && "text-destructive",
                    )}
                  >
                    {formatScoreDelta(task.score_delta) || "—"}
                  </TableCell>
                  <TableCell className="font-mono text-sm tabular-nums">{task.attempt ?? "—"}</TableCell>
                  <TableCell className="pr-4">
                    {task.wandb_url ? (
                      <a
                        href={task.wandb_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-mono text-xs text-primary underline-offset-2 hover:underline"
                      >
                        run
                      </a>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                    <CommandChips commands={task.commands} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
