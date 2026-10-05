#!/usr/bin/env bash
# Compare agents on two shared task sets: procedural patches and a folder of Vital presets.
# Usage: scripts/compare.sh [N] [PRESET_DIR]   (models are read from the env vars below)
set -uo pipefail
cd "$(dirname "$0")/.."
N=${1:-30}
PRESETS=${2:-$HOME/Music/Vital}
PY=${PY:-.venv/bin/python}
QWEN25_3B=${QWEN25_3B:-Qwen/Qwen2.5-Omni-3B}
QWEN25_7B=${QWEN25_7B:-Qwen/Qwen2.5-Omni-7B}
QWEN3=${QWEN3:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
GEMINI=${GEMINI:-gemini-3.8-flash}

bench() { "$PY" -m synth_rl.cli bench --specs mono --n "$N" "$@"; }

for set in procedural presets; do
  out=outputs/compare_$set
  src=$([ "$set" = procedural ] && echo procedural || echo "$PRESETS")
  bench --presets "$src" --out "$out" --agents rule
  if [ -n "${GEMINI_API_KEY:-}" ]; then
    bench --presets "$src" --out "$out" --agents "$GEMINI"
    bench --presets "$src" --out "$out" --agents "$GEMINI" --mute-audio
  fi
  bench --presets "$src" --out "$out" --agents "$QWEN25_3B" "$QWEN25_7B"
  bench --presets "$src" --out "$out" --agents "$QWEN25_7B" --mute-audio
  bench --presets "$src" --out "$out" --agents "$QWEN3"
done
