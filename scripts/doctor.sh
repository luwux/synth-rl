#!/usr/bin/env bash
# Collect everything needed to diagnose a failed or stalled run into one file.
# Usage: scripts/doctor.sh [RUN_DIR]   (default: the newest folder under runs/)
cd "$(dirname "$0")/.."
RUN=${1:-$(ls -td runs/*/ 2>/dev/null | head -1)}
RUN=${RUN%/}
OUT=${RUN:-.}/doctor.txt
PY=.venv/bin/python
{
  echo "== system"; date; uname -a; head -2 /etc/os-release 2>/dev/null
  echo "== gpus"; nvidia-smi 2>&1 | head -40
  echo "== disk"; df -h . | tail -1
  echo "== python"
  $PY -c "import sys, torch, transformers, peft
print(sys.version)
print('torch', torch.__version__, 'built for cuda', torch.version.cuda, 'available', torch.cuda.is_available(),
      'gpus', torch.cuda.device_count())
print('transformers', transformers.__version__, 'peft', peft.__version__)" 2>&1 | tail -6
  echo "== network (HTTP status; 000 = unreachable)"
  for url in https://huggingface.co ${HF_ENDPOINT:-} https://pypi.org/simple/ ${UV_DEFAULT_INDEX:-}; do
    echo "$url $(curl -s -o /dev/null -m 10 -w '%{http_code}' "$url")"
  done
  echo "== environment"; env | grep -E '^(CUDA_VISIBLE_DEVICES|HF_ENDPOINT|HF_HOME|UV_DEFAULT_INDEX|PIP_INDEX_URL|NCCL_)'
  echo "== processes"; ps -eo pid,etime,args | grep -E 'synth_rl|train\.sh' | grep -v grep
  if [ -n "$RUN" ]; then
    echo "== $RUN"; ls -la "$RUN" "$RUN/train" 2>&1
    echo "== preflight checks"; grep -E '^(PASS|FAIL)|preflight passed' "$RUN/preflight.log" 2>&1
    # Skip the table of unused speech-decoder weights that transformers prints when loading.
    for f in preflight.log run.log train/status.json; do
      echo "== $f (last 60 lines)"; grep -vE '\| (UNEXPECTED|MISSING) +\|' "$RUN/$f" 2>&1 | tail -60
    done
    echo "== errors in run.log"; grep -n -B2 -A8 -E 'Traceback|Error|out of memory' "$RUN/run.log" 2>/dev/null | tail -80
    echo "== train/log.jsonl (last 5 lines)"; tail -5 "$RUN/train/log.jsonl" 2>&1
  fi
} > "$OUT" 2>&1
echo "wrote $OUT"
