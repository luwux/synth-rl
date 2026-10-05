#!/usr/bin/env bash
# One command from a fresh machine to a finished training run:
#   1. Python environment (uv) with the pinned training dependencies
#   2. model download from Hugging Face
#   3. preflight checks; stops here on any failure
#   4. GRPO training in the background (scripts/train.sh): automatic restart and resume,
#      then a before/after benchmark and a results archive
#
# Usage: scripts/run_all.sh [options] [-- extra synth_rl.grpo arguments]
#   --model ID         Hugging Face model id (default Qwen/Qwen2.5-Omni-7B)
#   --gpus N           GPUs to train on (default: all visible; CUDA_VISIBLE_DEVICES=0,1,2,3 selects cards,
#                      so two runs can share a machine)
#   --hours H          wall-time budget for training and evaluation (default 20)
#   --steps N          maximum GRPO steps (default 400)
#   --name NAME        run folder under runs/ (default: <model>-<date>)
#   --skip-preflight   go straight to training
#   --foreground       train in this shell instead of detaching
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=Qwen/Qwen2.5-Omni-7B
GPUS=""
HOURS=20
STEPS=400
NAME=""
PREFLIGHT=1
FOREGROUND=0
while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --hours) HOURS=$2; shift 2 ;;
    --steps) STEPS=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --skip-preflight) PREFLIGHT=0; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    --) shift; break ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
if [ -z "$GPUS" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
  else
    GPUS=$(command -v nvidia-smi >/dev/null && nvidia-smi -L | wc -l || echo 1)
  fi
fi
NAME=${NAME:-$(basename "$MODEL")-$(date +%Y%m%d-%H%M)}
RUN=runs/$NAME
mkdir -p "$RUN"
step() { echo; echo "== $* ($(date +%H:%M:%S))"; }

step "1/4 environment"
if ! command -v uv >/dev/null; then
  # pip follows PIP_INDEX_URL mirrors; the official installer downloads from GitHub
  python3 -m pip install --user -q uv || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
# Python 3.12 (tested); if it is neither installed nor downloadable, any installed 3.10+
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv || uv venv --python ">=3.10" .venv
uv pip install --python .venv/bin/python -e ".[train]"
PY=.venv/bin/python
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
if command -v nvidia-smi >/dev/null && ! $PY -c "import torch, sys; sys.exit(not torch.cuda.is_available())"; then
  echo "nvidia-smi sees GPUs but PyTorch cannot use them; the NVIDIA driver is probably older than this PyTorch build needs:"
  nvidia-smi | head -4
  exit 1
fi

step "2/4 model $MODEL"
MODEL_DIR=models/$(basename "$MODEL")
if [ -d "$MODEL" ]; then
  MODEL_DIR=$MODEL
else
  .venv/bin/hf download "$MODEL" --local-dir "$MODEL_DIR"
fi

if [ "$PREFLIGHT" = 1 ]; then
  step "3/4 preflight on $GPUS GPU(s)"
  $PY -m synth_rl.preflight --model "$MODEL_DIR" --gpus "$GPUS" --out "$RUN/preflight" 2>&1 | tee "$RUN/preflight.log"
fi

step "4/4 training"
cmd=(scripts/train.sh "$MODEL_DIR" "$RUN" "$GPUS" "$HOURS" "$STEPS" "$@")
if [ "$FOREGROUND" = 1 ]; then
  "${cmd[@]}" 2>&1 | tee "$RUN/run.log"
else
  nohup "${cmd[@]}" > "$RUN/run.log" 2>&1 &
  echo "training started in the background (pid $!)."
  echo "  progress:  tail -f $RUN/run.log"
  echo "  metrics:   $RUN/train/log.jsonl   samples: $RUN/train/samples.jsonl"
  echo "  results:   $RUN/results.tar.gz when finished"
  echo "  resume after a crash or reboot: scripts/train.sh ${cmd[*]:1}"
fi
