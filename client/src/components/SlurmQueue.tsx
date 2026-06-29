import type { SlurmJob } from "../types/status";
import { formatDuration, formatEtaLabel } from "../utils/format";
import { CommandChips } from "./CommandChips";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface SlurmQueueProps {
  jobs: SlurmJob[];
}

function formatPosition(job: SlurmJob): string {
  if (job.state === "RUNNING") return "·";
  if (job.queue_position != null) return String(job.queue_position).padStart(2, "0");
  return "—";
}

function StateBadge({ state }: { state: string }) {
  if (state === "RUNNING") {
    return (
      <Badge className="bg-status-running/15 text-status-running hover:bg-status-running/15">
        running
      </Badge>
    );
  }
  if (state === "PENDING") {
    return (
      <Badge variant="outline" className="border-status-pending/40 text-status-pending">
        pending
      </Badge>
    );
  }
  return <Badge variant="secondary">{state.toLowerCase()}</Badge>;
}

const ETA_HINTS: Record<string, string> = {
  time_limit_remaining: "Remaining until Slurm TIME_LIMIT, not when your job actually ends.",
  scheduled_start: "Slurm scheduled start estimate.",
  reported: "ETA reported by job_progress.report().",
  extrapolated: "Linear extrapolation from step/total and elapsed time.",
  stale_progress: "No heartbeat recently; progress may be stale.",
};

export function SlurmQueue({ jobs }: SlurmQueueProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Cluster queue</CardTitle>
        <CardDescription>
          Running first, then pending by priority. &ldquo;Limit left&rdquo; is worst case until
          TIME_LIMIT, not real finish time.
        </CardDescription>
      </CardHeader>
      <CardContent className="px-0">
        {jobs.length === 0 ? (
          <p className="px-4 text-sm text-muted-foreground">No active jobs on the account.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="w-10 pl-4 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  #
                </TableHead>
                <TableHead className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  Job
                </TableHead>
                <TableHead className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  Who
                </TableHead>
                <TableHead className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  GPU
                </TableHead>
                <TableHead className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  State
                </TableHead>
                <TableHead className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  Pri
                </TableHead>
                <TableHead className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  Elapsed
                </TableHead>
                <TableHead className="pr-4 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  Clock
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.map((job) => {
                const progressPct = job.progress?.progress_pct ?? null;
                const limitPct =
                  progressPct == null &&
                  job.elapsed_seconds != null &&
                  job.time_limit_seconds != null &&
                  job.time_limit_seconds > 0
                    ? Math.min(100, (job.elapsed_seconds / job.time_limit_seconds) * 100)
                    : null;
                const etaHint = job.eta_kind ? ETA_HINTS[job.eta_kind] ?? "" : "";

                return (
                  <TableRow key={job.job_id} className="align-top">
                    <TableCell className="pl-4 font-mono text-sm text-muted-foreground tabular-nums">
                      {formatPosition(job)}
                    </TableCell>
                    <TableCell>
                      <div className="font-mono text-sm font-medium text-foreground">{job.job_id}</div>
                      <div className="mt-0.5 max-w-[15rem] truncate text-xs text-muted-foreground">
                        {job.name}
                      </div>
                      {job.compatible_with.length > 0 && (
                        <p className="mt-1.5 font-mono text-[10px] text-muted-foreground">
                          shares node: {job.compatible_with.join(", ")}
                        </p>
                      )}
                      {job.progress?.message && (
                        <p className="mt-1 font-mono text-[10px] text-muted-foreground">
                          {job.progress.phase}: {job.progress.message}
                          {job.progress.heartbeat_stale ? " (stale)" : ""}
                        </p>
                      )}
                      {progressPct != null && <Progress value={progressPct} className="mt-2 h-1.5" />}
                      {limitPct != null && (
                        <Progress
                          value={limitPct}
                          className="mt-2 h-1 [&_[data-slot=progress-indicator]]:bg-muted-foreground/50"
                        />
                      )}
                      <CommandChips commands={job.commands} />
                    </TableCell>
                    <TableCell className="text-sm">{job.user}</TableCell>
                    <TableCell className="font-mono text-sm tabular-nums">{job.gpus ?? "—"}</TableCell>
                    <TableCell>
                      <StateBadge state={job.state} />
                      {job.state === "PENDING" && job.reason && (
                        <p className="mt-1 max-w-[9rem] text-[10px] leading-snug text-muted-foreground">
                          {job.reason}
                        </p>
                      )}
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground tabular-nums">
                      {job.priority ?? "—"}
                    </TableCell>
                    <TableCell className="font-mono text-sm tabular-nums">
                      {formatDuration(job.elapsed_seconds)}
                    </TableCell>
                    <TableCell className="pr-4">
                      {job.eta_seconds != null ? (
                        <Tooltip>
                          <TooltipTrigger>
                            <div className="cursor-default">
                              <p className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                                {formatEtaLabel(job.eta_kind)}
                              </p>
                              <p className="font-mono text-sm tabular-nums">
                                {formatDuration(job.eta_seconds)}
                              </p>
                            </div>
                          </TooltipTrigger>
                          {etaHint && (
                            <TooltipContent side="left" className="max-w-[240px] text-xs">
                              {etaHint}
                            </TooltipContent>
                          )}
                        </Tooltip>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
