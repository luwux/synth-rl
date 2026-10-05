#!/usr/bin/env bash
# Listening-dependent comparisons on procedural patches: blind edits (values hidden), repair, and
# from-scratch matching.
# Usage: scripts/compare_listening.sh [N]   (models from the same env vars as compare.sh)
set -uo pipefail
cd "$(dirname "$0")/.."
N=${1:-30}
PY=${PY:-.venv/bin/python}
QWEN25_3B=${QWEN25_3B:-Qwen/Qwen2.5-Omni-3B}
QWEN25_7B=${QWEN25_7B:-Qwen/Qwen2.5-Omni-7B}
QWEN3=${QWEN3:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
GEMINI=${GEMINI:-gemini-3.8-flash}

for mode in blind repair match; do
  out=outputs/compare_$mode
  flags=(--specs mono --n "$N" --out "$out")
  if [ "$mode" = blind ]; then flags+=(--blind); else flags+=(--kind "$mode"); fi
  "$PY" -m synth_rl.cli bench "${flags[@]}" --agents rule
  if [ -n "${GEMINI_API_KEY:-}" ]; then
    "$PY" -m synth_rl.cli bench "${flags[@]}" --agents "$GEMINI"
    "$PY" -m synth_rl.cli bench "${flags[@]}" --agents "$GEMINI" --mute-audio
  fi
  "$PY" -m synth_rl.cli bench "${flags[@]}" --agents "$QWEN25_3B" "$QWEN25_7B"
  "$PY" -m synth_rl.cli bench "${flags[@]}" --agents "$QWEN25_7B" --mute-audio
  "$PY" -m synth_rl.cli bench "${flags[@]}" --agents "$QWEN3"
done
