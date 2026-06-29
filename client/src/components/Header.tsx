import { Bell, BellOff, RefreshCw } from "lucide-react";
import type { DashboardStatus } from "../types/status";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface HeaderProps {
  status: DashboardStatus | null;
  onRefresh: () => void;
  notifyEnabled?: boolean;
  onToggleNotify?: () => void;
}

export function Header({ status, onRefresh, notifyEnabled, onToggleNotify }: HeaderProps) {
  const isMock = status?.mode === "mock";
  const time = status?.timestamp
    ? new Date(status.timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : "—";

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/85 backdrop-blur supports-[backdrop-filter]:bg-background/70">
      <div className="mx-auto flex h-14 max-w-[1400px] items-center justify-between gap-4 px-6">
        <div className="flex items-center gap-3">
          <span className="text-lg font-semibold tracking-tight text-foreground">LOKI</span>
          <span className="h-4 w-px bg-border" aria-hidden />
          <span className="hidden text-sm text-muted-foreground sm:inline">CISPA Finals</span>
          <span className="rounded-md bg-muted px-2 py-0.5 font-mono text-xs text-muted-foreground">
            {status?.account ?? "—"}
          </span>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden items-center gap-2 md:flex">
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
                isMock
                  ? "bg-status-mock/15 text-status-pending"
                  : "bg-status-running/15 text-status-running",
              )}
            >
              <span
                className={cn(
                  "size-1.5 animate-pulse rounded-full",
                  isMock ? "bg-status-pending" : "bg-status-running",
                )}
              />
              {status?.mode ?? "…"}
            </span>
            <span className="font-mono text-xs text-muted-foreground tabular-nums">
              {time}
              <span className="mx-1.5 text-border">·</span>
              {status?.refresh_seconds ?? "—"}s
            </span>
          </div>

          {onToggleNotify && (
            <Button
              variant="outline"
              size="sm"
              onClick={onToggleNotify}
              aria-pressed={notifyEnabled}
              title={notifyEnabled ? "Alerts on" : "Alerts off"}
            >
              {notifyEnabled ? <Bell /> : <BellOff />}
              <span className="hidden sm:inline">Alerts</span>
            </Button>
          )}
          <Button size="sm" onClick={onRefresh}>
            <RefreshCw />
            Sync
          </Button>
        </div>
      </div>
    </header>
  );
}
