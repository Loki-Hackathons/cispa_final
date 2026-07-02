#!/usr/bin/env python3
"""Local browser viewer for Task 1 labeled splits (train / validation).

Ground truth only — no predictions required. Browse all documents with
watermarked spans highlighted.

Usage (from cispa_final root):
    python scripts/task1/view_dataset.py
    python scripts/task1/view_dataset.py --dataset-dir data/watermark_localization --port 8765

Then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "watermark_localization"

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Task 1 — dataset viewer</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
           margin: 0; background: #0f1117; color: #e5e7eb; }
    header { padding: 1rem 1.25rem; border-bottom: 1px solid #2a2f3a; background: #151922; }
    h1 { margin: 0 0 .35rem; font-size: 1.1rem; font-weight: 600; }
    .sub { color: #9ca3af; font-size: .85rem; }
    .bar { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin-top: .75rem; }
    select, button { background: #1f2430; color: #e5e7eb; border: 1px solid #374151;
                     border-radius: 6px; padding: .4rem .6rem; font: inherit; }
    button { cursor: pointer; }
    button:hover { background: #2a3140; }
    main { padding: 1.25rem; max-width: 1100px; margin: 0 auto; }
    .legend { display: flex; flex-wrap: wrap; gap: 1rem; font-size: .75rem; color: #9ca3af;
              margin-bottom: 1rem; text-transform: uppercase; letter-spacing: .04em; }
    .chip { display: inline-flex; align-items: center; gap: .35rem; }
    .chip i { display: inline-block; width: 1.1rem; height: .65rem; border-radius: 2px; }
    .wm { background: rgba(34,197,94,.35); border-bottom: 2px solid #22c55e; }
    .clean { background: transparent; }
    .text { line-height: 1.9; font-size: .95rem; background: #151922; border: 1px solid #2a2f3a;
            border-radius: 10px; padding: 1rem 1.1rem; min-height: 120px; white-space: pre-wrap; }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
             gap: .6rem; margin: 1rem 0; }
    .stat { background: #151922; border: 1px solid #2a2f3a; border-radius: 8px; padding: .6rem .75rem; }
    .stat label { display: block; color: #9ca3af; font-size: .7rem; text-transform: uppercase; }
    .stat span { font-size: 1.1rem; }
    .preview { color: #9ca3af; font-size: .85rem; margin-top: .75rem; max-height: 4.5em; overflow: hidden; }
  </style>
</head>
<body>
  <header>
    <h1>Task 1 — dataset viewer (local)</h1>
    <div class="sub">Ground truth: green highlight + underline = watermarked token</div>
    <div class="bar">
      <label>Split <select id="split"></select></label>
      <label>Document <select id="doc"></select></label>
      <button id="prev" type="button">← Prev</button>
      <button id="next" type="button">Next →</button>
    </div>
  </header>
  <main>
    <div class="legend">
      <span class="chip"><i class="wm"></i> watermarked (label=1)</span>
      <span class="chip"><i class="clean"></i> clean (label=0)</span>
    </div>
    <div class="stats">
      <div class="stat"><label>Tokens</label><span id="nTokens">—</span></div>
      <div class="stat"><label>Watermarked</label><span id="nWm">—</span></div>
      <div class="stat"><label>Clean</label><span id="nClean">—</span></div>
      <div class="stat"><label>WM ratio</label><span id="wmRatio">—</span></div>
    </div>
    <div class="text" id="tokens"></div>
    <div class="preview" id="preview"></div>
  </main>
  <script>
    let catalog = {};
    let docs = [];
    let idx = 0;

    function decodePiece(piece) {
      if (piece.includes("Ċ")) return { text: piece.split("Ċ").join(""), newline: true };
      return { text: piece.split("Ġ").join(" "), newline: false };
    }

    function renderDoc(doc) {
      const nWm = doc.labels.reduce((a, b) => a + b, 0);
      document.getElementById("nTokens").textContent = doc.labels.length;
      document.getElementById("nWm").textContent = nWm;
      document.getElementById("nClean").textContent = doc.labels.length - nWm;
      document.getElementById("wmRatio").textContent = (100 * nWm / doc.labels.length).toFixed(1) + "%";
      document.getElementById("preview").textContent = doc.text || "";
      const root = document.getElementById("tokens");
      root.textContent = "";
      doc.token_pieces.forEach((piece, i) => {
        const { text, newline } = decodePiece(piece);
        const span = document.createElement("span");
        span.className = doc.labels[i] === 1 ? "wm" : "clean";
        span.title = `label=${doc.labels[i]}`;
        span.textContent = text || "·";
        root.appendChild(span);
        if (newline) root.appendChild(document.createElement("br"));
      });
    }

    function fillDocSelect() {
      const sel = document.getElementById("doc");
      sel.innerHTML = "";
      docs.forEach((d, i) => {
        const wm = d.labels.reduce((a, b) => a + b, 0);
        const opt = document.createElement("option");
        opt.value = i;
        opt.textContent = `${d.document_id} (${d.labels.length} tok, ${wm} wm)`;
        sel.appendChild(opt);
      });
      idx = 0;
      sel.value = "0";
      renderDoc(docs[0]);
    }

    async function loadSplit(split) {
      const res = await fetch(`/api/documents?split=${split}`);
      docs = await res.json();
      fillDocSelect();
    }

    async function init() {
      const res = await fetch("/api/splits");
      const splits = await res.json();
      const sel = document.getElementById("split");
      splits.forEach(s => {
        const opt = document.createElement("option");
        opt.value = s.name;
        opt.textContent = `${s.name} (${s.count} docs)`;
        sel.appendChild(opt);
      });
      sel.addEventListener("change", () => loadSplit(sel.value));
      document.getElementById("doc").addEventListener("change", e => {
        idx = Number(e.target.value);
        renderDoc(docs[idx]);
      });
      document.getElementById("prev").addEventListener("click", () => {
        idx = (idx - 1 + docs.length) % docs.length;
        document.getElementById("doc").value = String(idx);
        renderDoc(docs[idx]);
      });
      document.getElementById("next").addEventListener("click", () => {
        idx = (idx + 1) % docs.length;
        document.getElementById("doc").value = String(idx);
        renderDoc(docs[idx]);
      });
      await loadSplit(splits[0].name);
    }
    init();
  </script>
</body>
</html>
"""


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def slim(doc: dict) -> dict:
    return {
        "document_id": doc["document_id"],
        "text": doc.get("text", ""),
        "token_pieces": doc["token_pieces"],
        "labels": doc["labels"],
    }


def make_handler(store: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/splits":
                payload = [{"name": k, "count": len(v)} for k, v in store.items()]
                self._send(200, json.dumps(payload).encode(), "application/json")
                return
            if parsed.path == "/api/documents":
                qs = parse_qs(parsed.query)
                split = qs.get("split", ["train"])[0]
                if split not in store:
                    self._send(404, b'{"error":"unknown split"}', "application/json")
                    return
                self._send(200, json.dumps(store[split]).encode(), "application/json")
                return
            self._send(404, b"not found", "text/plain")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Task 1 dataset viewer")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET,
        help="Folder with train.jsonl and validation.jsonl",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "validation.jsonl"
    for p in (train_path, val_path):
        if not p.is_file():
            raise SystemExit(f"Missing {p} — clone the dataset first (see data/README.md)")

    store = {
        "train": [slim(d) for d in load_jsonl(train_path)],
        "validation": [slim(d) for d in load_jsonl(val_path)],
    }
    n = sum(len(v) for v in store.values())
    print(f"Loaded {n} labeled documents from {dataset_dir}")

    url = f"http://{args.host}:{args.port}"
    server = HTTPServer((args.host, args.port), make_handler(store))
    print(f"Open {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
