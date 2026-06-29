#!/bin/bash
# Build React client and print run instructions.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/client"

echo "Installing client dependencies..."
npm install

echo "Building client..."
npm run build

echo ""
echo "Done. Start the dashboard:"
echo "  cd $ROOT"
echo "  # Edit dashboard/config.py → MODE = 'mock' or 'live'"
echo "  python -m uvicorn dashboard.server:app --host 127.0.0.1 --port 8080"
echo ""
echo "Windows: .\\scripts\\run_dashboard.ps1"
echo ""
echo "Browser: http://127.0.0.1:8080"
echo "SSH tunnel: ssh -L 8080:localhost:8080 <user>@jureca.fz-juelich.de"
