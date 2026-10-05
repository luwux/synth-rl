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
#   --hours H          wall-time budget for the whole run from now: install, download, checks, training,
#                      and the final benchmark (default 20)
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
# scripts/train.sh keeps this deadline across restarts
[ -f "$RUN/deadline" ] || python3 -c "import sys, time; print(int(time.time() + float(sys.argv[1]) * 3600))" \
  "$HOURS" > "$RUN/deadline"
step() { echo; echo "== $* ($(date +%H:%M:%S))"; }

# Behind a restricted network, switch to mirrors unless an index or endpoint was chosen already.
reachable() { curl -s -o /dev/null -m 8 "$1"; }
if [ -z "${HF_ENDPOINT:-}" ] && ! reachable https://huggingface.co; then
  export HF_ENDPOINT=https://hf-mirror.com
  echo "huggingface.co unreachable; downloading models from $HF_ENDPOINT"
fi
if [ -z "${UV_DEFAULT_INDEX:-}${PIP_INDEX_URL:-}" ] && ! reachable https://pypi.org/simple/; then
  export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
  echo "pypi.org unreachable; installing packages from $UV_DEFAULT_INDEX"
fi

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
cuda_ok() { $PY -c "import torch, sys; sys.exit(not torch.cuda.is_available())"; }
if command -v nvidia-smi >/dev/null && ! cuda_ok; then
  # The PyPI build bundles CUDA 13 and needs NVIDIA driver 580 or newer; the CUDA 12.6 build runs on 525 and newer.
  TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
  echo "nvidia-smi sees GPUs but PyTorch cannot use them; the NVIDIA driver is probably older than its CUDA build needs."
  nvidia-smi | head -4
  echo "installing the same PyTorch built for ${TORCH_INDEX##*/}"
  read -r TV VV < <($PY -c "from importlib.metadata import version as v; print(v('torch').split('+')[0], v('torchvision').split('+')[0])")
  uv pip install --python $PY --extra-index-url "$TORCH_INDEX" "torch==$TV+${TORCH_INDEX##*/}" "torchvision==$VV+${TORCH_INDEX##*/}"
  if ! cuda_ok; then
    echo "PyTorch still cannot use the GPUs; update the NVIDIA driver, or set TORCH_INDEX to another build"
    exit 1
  fi
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
  if ! $PY -m synth_rl.preflight --model "$MODEL_DIR" --gpus "$GPUS" --out "$RUN/preflight" 2>&1 | tee "$RUN/preflight.log"; then
    echo "preflight failed; scripts/doctor.sh $RUN collects the details (see AGENTS.md)"; exit 1
  fi
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
  echo "  diagnose a failure: scripts/doctor.sh $RUN"
  echo "  stop (the restart loop first, then its processes): pkill -f \"train.sh .*$RUN \"; pkill -f $RUN/"
fi
