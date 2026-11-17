#!/bin/zsh
# Start the demo server (and, with --vital, the real Vital window that mirrors it). Open http://127.0.0.1:8765
# Ctrl-C stops everything it started.
cd "$(dirname "$0")/.."
[[ -d demo/web/dist ]] || (cd demo/web && pnpm install && pnpm build)
pids=()
trap 'kill $pids 2>/dev/null; wait' EXIT INT TERM
.venv/bin/uvicorn demo.server.app:app --host 127.0.0.1 --port 8765 &
pids+=($!)
if [[ $1 == --vital ]]; then
  until curl -s -o /dev/null http://127.0.0.1:8765/api/state; do sleep 0.3; done
  uv run -q --no-project --python 3.12 --with pedalboard python demo/mirror/vital_mirror.py &
  pids+=($!)
fi
wait $pids[1]
