import { useEffect, useRef } from "react";
import type { DashboardStatus } from "../types/status";

function notify(title: string, body: string) {
  if (typeof Notification === "undefined") return;
  if (Notification.permission !== "granted") return;
  new Notification(title, { body });
}

export function useNotifications(data: DashboardStatus | null, enabled: boolean) {
  const prev = useRef<DashboardStatus | null>(null);

  useEffect(() => {
    if (!enabled || !data) return;

    if (typeof Notification !== "undefined" && Notification.permission === "default") {
      void Notification.requestPermission();
    }

    const prior = prev.current;
    prev.current = data;
    if (!prior) return;

    for (const task of data.tasks) {
      const old = prior.tasks.find((t) => t.task_id === task.task_id);
      if (!old) continue;
      if (!old.submit_ready && task.submit_ready) {
        notify("Submit gate open", `${task.owner ?? "team"}: ${task.task_id}`);
      }
      if (!old.query_ready && task.query_ready) {
        notify("Query gate open", `${task.owner ?? "team"}: ${task.task_id}`);
      }
    }

    if (data.failed_jobs.length > prior.failed_jobs.length) {
      const latest = data.failed_jobs[0];
      if (latest) {
        notify("Job failed", `${latest.user} ${latest.job_id} ${latest.state}`);
      }
    }
  }, [data, enabled]);
}
