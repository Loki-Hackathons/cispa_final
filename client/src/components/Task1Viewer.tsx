import { useEffect, useMemo, useState } from "react";
import { fetchTask1Attempt, fetchTask1Attempts } from "../api/client";
import type { Task1AttemptSummary, Task1Bundle } from "../types/status";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";

function decodeTokenPiece(piece: string): { text: string; newline: boolean } {
  if (piece.includes("Ċ")) return { text: piece.split("Ċ").join(""), newline: true };
  return { text: piece.split("Ġ").join(" "), newline: false };
}

// score in [0,1] -> background color from near-transparent to red
function confidenceColor(score: number): string {
  const alpha = 0.08 + score * 0.72;
  return `rgba(239, 68, 68, ${alpha.toFixed(3)})`;
}

export function Task1Viewer() {
  const [attempts, setAttempts] = useState<Task1AttemptSummary[]>([]);
  const [attemptId, setAttemptId] = useState<string | null>(null);
  const [bundle, setBundle] = useState<Task1Bundle | null>(null);
  const [docIndex, setDocIndex] = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchTask1Attempts()
      .then((data) => {
        setAttempts(data);
        if (data.length > 0) setAttemptId(data[0].id);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!attemptId) return;
    setLoading(true);
    setDocIndex(0);
    fetchTask1Attempt(attemptId)
      .then(setBundle)
      .catch(() => setBundle(null))
      .finally(() => setLoading(false));
  }, [attemptId]);

  const doc = bundle?.documents[docIndex];

  const stats = useMemo(() => {
    if (!doc) return null;
    let tp = 0, fp = 0, fn = 0, tn = 0;
    doc.labels.forEach((label, i) => {
      const predicted = doc.scores[i] >= 0.5;
      const actual = label === 1;
      if (predicted && actual) tp++;
      else if (predicted && !actual) fp++;
      else if (!predicted && actual) fn++;
      else tn++;
    });
    return { tp, fp, fn, tn };
  }, [doc]);

  if (attempts.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Task 1 — Token viewer</CardTitle>
          <CardDescription>
            No local eval bundles yet. Run{" "}
            <code className="font-mono text-primary">shared/task1_eval.py</code> after scoring
            the validation set.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Task 1 — Token viewer</CardTitle>
        <CardDescription>
          Ground truth (underline) vs our predicted confidence (background). Hover a token for
          the exact score.
        </CardDescription>
        <div className="flex flex-wrap items-center gap-2 pt-2">
          <select
            value={attemptId ?? ""}
            onChange={(e) => setAttemptId(e.target.value)}
            className="rounded border border-border bg-muted px-2 py-1 font-mono text-xs"
          >
            {attempts.map((a) => (
              <option key={a.id} value={a.id}>
                {a.id} — {a.method ?? "no method"} (TPR {a["tpr_at_0.1pct_fpr"]?.toFixed(3) ?? "—"})
              </option>
            ))}
          </select>
          {bundle && bundle.documents.length > 1 && (
            <select
              value={docIndex}
              onChange={(e) => setDocIndex(Number(e.target.value))}
              className="rounded border border-border bg-muted px-2 py-1 font-mono text-xs"
            >
              {bundle.documents.map((d, i) => (
                <option key={d.document_id} value={i}>
                  {d.document_id}
                </option>
              ))}
            </select>
          )}
        </div>
        {bundle?.note && (
          <p className="pt-1 text-xs text-muted-foreground">Note: {bundle.note}</p>
        )}
      </CardHeader>
      <CardContent>
        {loading && <p className="text-sm text-muted-foreground">Loading…</p>}
        {!loading && doc && (
          <>
            <div className="mb-3 flex flex-wrap gap-3 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              <span className="flex items-center gap-1">
                <span
                  className="inline-block h-2.5 w-4 rounded-sm border-b-2 border-status-running"
                  style={{ backgroundColor: confidenceColor(0.1) }}
                />
                low confidence, watermarked (GT)
              </span>
              <span className="flex items-center gap-1">
                <span
                  className="inline-block h-2.5 w-4 rounded-sm"
                  style={{ backgroundColor: confidenceColor(0.9) }}
                />
                high confidence
              </span>
              <span className="border-b-2 border-status-running px-1">underline = watermarked (ground truth)</span>
            </div>

            <div className="rounded-lg border border-border bg-muted/30 p-4 font-mono text-sm leading-8">
              {doc.token_pieces.map((piece, i) => {
                const { text, newline } = decodeTokenPiece(piece);
                const label = doc.labels[i];
                const score = doc.scores[i];
                return (
                  <span key={i}>
                    <span
                      title={`score=${score.toFixed(4)}  label=${label === 1 ? "watermarked" : "clean"}`}
                      className={cn(
                        "rounded-sm px-0.5",
                        label === 1 && "border-b-2 border-status-running",
                      )}
                      style={{ backgroundColor: confidenceColor(score) }}
                    >
                      {text || "·"}
                    </span>
                    {newline && <br />}
                  </span>
                );
              })}
            </div>

            {stats && (
              <div className="mt-3 grid grid-cols-4 gap-2 font-mono text-xs">
                <div className="rounded border border-border p-2 text-center">
                  <div className="text-muted-foreground">TP</div>
                  <div className="text-status-running">{stats.tp}</div>
                </div>
                <div className="rounded border border-border p-2 text-center">
                  <div className="text-muted-foreground">FP</div>
                  <div className="text-destructive">{stats.fp}</div>
                </div>
                <div className="rounded border border-border p-2 text-center">
                  <div className="text-muted-foreground">FN</div>
                  <div className="text-status-pending">{stats.fn}</div>
                </div>
                <div className="rounded border border-border p-2 text-center">
                  <div className="text-muted-foreground">TN</div>
                  <div>{stats.tn}</div>
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
