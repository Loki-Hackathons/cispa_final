import type { FailedJob } from "../types/status";
import { CommandChips } from "./CommandChips";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface FailedJobsPanelProps {
  jobs: FailedJob[];
}

export function FailedJobsPanel({ jobs }: FailedJobsPanelProps) {
  if (jobs.length === 0) return null;

  return (
    <Card className="ring-destructive/20">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-destructive">
          <span className="size-2 rounded-full bg-destructive" aria-hidden />
          Recent failures
        </CardTitle>
        <CardDescription>Last {jobs.length} failed or timed-out jobs (sacct).</CardDescription>
      </CardHeader>
      <CardContent>
        <ul className="divide-y divide-border">
          {jobs.map((job) => (
            <li key={job.job_id} className="py-3 first:pt-0 last:pb-0">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="font-mono text-sm font-medium text-destructive">{job.job_id}</span>
                <span className="font-mono text-[10px] text-muted-foreground">{job.ended_at ?? "—"}</span>
              </div>
              <p className="mt-1 text-sm text-foreground">
                {job.user} · {job.name} · {job.state}
                {job.exit_code ? ` (${job.exit_code})` : ""}
              </p>
              {job.log_err && (
                <p className="mt-1 font-mono text-[10px] text-muted-foreground">{job.log_err}</p>
              )}
              <CommandChips commands={job.commands} />
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
