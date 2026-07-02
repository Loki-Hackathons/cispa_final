import type { LeaderboardRow } from "../types/status";
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

interface LeaderboardPanelProps {
  rows: LeaderboardRow[];
}

const HEAD = "font-mono text-[10px] uppercase tracking-wider text-muted-foreground";

export function LeaderboardPanel({ rows }: LeaderboardPanelProps) {
  if (rows.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Leaderboard</CardTitle>
        <CardDescription>
          Team ranks from dashboard state ·{" "}
          <a
            href="http://35.192.205.84/leaderboard_page"
            className="text-primary underline-offset-2 hover:underline"
            target="_blank"
            rel="noreferrer"
          >
            full leaderboard
          </a>
        </CardDescription>
      </CardHeader>
      <CardContent className="px-0">
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className={cn(HEAD, "pl-4")}>Task</TableHead>
              <TableHead className={HEAD}>Rank</TableHead>
              <TableHead className={HEAD}>Score</TableHead>
              <TableHead className={cn(HEAD, "pr-4")}>Gap</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.task_id}>
                <TableCell className="pl-4 font-mono text-sm">{row.task_id}</TableCell>
                <TableCell className="font-mono text-sm tabular-nums">{row.team_rank ?? "—"}</TableCell>
                <TableCell className="font-mono text-sm tabular-nums">
                  {row.team_score != null ? row.team_score.toFixed(4) : "—"}
                </TableCell>
                <TableCell className="pr-4 font-mono text-sm tabular-nums">
                  {row.gap != null ? row.gap.toFixed(4) : "—"}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
