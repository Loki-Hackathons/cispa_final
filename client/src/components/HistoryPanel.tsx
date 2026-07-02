import { useEffect, useState } from "react";
import { fetchHistory } from "../api/client";
import type { HistoryEvent } from "../types/status";
import {
  Card,
  CardContent,
  CardDescription,
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

const HEAD = "font-mono text-[10px] uppercase tracking-wider text-muted-foreground";
const KIND_COLORS: Record<string, string> = {
  submit: "text-status-running",
  analysis: "text-primary",
  logits: "text-muted-foreground",
  experiment: "text-status-pending",
};

function shortFile(path?: string): string {
  if (!path) return "";
  const parts = path.split(/[\\/]/);
  return parts.length > 2 ? `…/${parts.slice(-2).join("/")}` : path;
}

export function HistoryPanel() {
  const [events, setEvents] = useState<HistoryEvent[]>([]);
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    let active = true;
    const load = () =>
      fetchHistory(200)
        .then((data) => active && setEvents(data))
        .catch(() => undefined);
    load();
    const id = setInterval(load, 15000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  const kinds = ["all", ...Array.from(new Set(events.map((e) => e.kind)))];
  const shown = filter === "all" ? events : events.filter((e) => e.kind === filter);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Submission history</CardTitle>
        <CardDescription>
          Every submit / analysis / experiment, newest first ({events.length} events).
        </CardDescription>
        <div className="flex flex-wrap gap-1 pt-1">
          {kinds.map((k) => (
            <button
              key={k}
              onClick={() => setFilter(k)}
              className={cn(
                "rounded border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
                filter === k
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border text-muted-foreground hover:bg-muted",
              )}
            >
              {k}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent className="max-h-80 overflow-y-auto px-0">
        {shown.length === 0 ? (
          <p className="px-4 text-sm text-muted-foreground">No events yet.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className={cn(HEAD, "pl-4")}>Time</TableHead>
                <TableHead className={HEAD}>Kind</TableHead>
                <TableHead className={HEAD}>Task</TableHead>
                <TableHead className={HEAD}>Owner</TableHead>
                <TableHead className={HEAD}>Score</TableHead>
                <TableHead className={cn(HEAD, "pr-4")}>File / note</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {shown.map((ev, i) => (
                <TableRow key={`${ev.ts}-${i}`} className="align-top">
                  <TableCell className="pl-4 font-mono text-xs tabular-nums text-muted-foreground">
                    {ev.ts.replace("T", " ")}
                  </TableCell>
                  <TableCell
                    className={cn("font-mono text-xs uppercase", KIND_COLORS[ev.kind] ?? "")}
                  >
                    {ev.kind}
                  </TableCell>
                  <TableCell className="font-mono text-xs">{ev.task_id ?? "—"}</TableCell>
                  <TableCell className="font-mono text-xs">{ev.owner ?? "—"}</TableCell>
                  <TableCell className="font-mono text-xs font-medium tabular-nums">
                    {ev.score != null ? ev.score.toFixed(6) : "—"}
                  </TableCell>
                  <TableCell className="pr-4 text-xs">
                    <span className="font-mono text-muted-foreground" title={ev.file}>
                      {shortFile(ev.file)}
                    </span>
                    {ev.note && <p className="mt-0.5 text-foreground">{ev.note}</p>}
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
