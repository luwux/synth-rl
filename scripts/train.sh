#!/usr/bin/env bash
# GRPO training with automatic restart (every restart resumes from the last checkpoint), then a
# before/after benchmark on fixed task sets and a results archive. Normally started by scripts/run_all.sh.
# Usage: scripts/train.sh MODEL_DIR RUN_DIR GPUS HOURS STEPS [extra synth_rl.grpo arguments]
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=$1 RUN=$2 GPUS=$3 HOURS=$4 STEPS=$5
shift 5
PY=${PY:-.venv/bin/python}
EVAL_N=${EVAL_N:-100}
EVAL_RESERVE=${EVAL_RESERVE:-5400}  # seconds kept for the final benchmark
# Fewer out-of-memory errors from fragmentation while sequence lengths vary
command -v nvidia-smi >/dev/null && [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ] \
  && export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
if [ "$GPUS" -gt 1 ]; then
  LAUNCH=("$PY" -m torch.distributed.run --standalone --nproc_per_node "$GPUS")
else
  LAUNCH=("$PY")
fi
BLIND=()  # a policy trained on hidden parameter values is benchmarked the same way
for a in "$@"; do [ "$a" = --blind ] && BLIND=(--blind); done
# The deadline survives restarts of this script.
[ -f "$RUN/deadline" ] || $PY -c "import time; print(int(time.time() + $HOURS * 3600))" > "$RUN/deadline"
deadline=$(cat "$RUN/deadline")

attempt=0
while true; do
  left=$((deadline - $(date +%s) - EVAL_RESERVE))
  if [ "$left" -le 300 ]; then echo "training time budget used"; break; fi
  hours=$($PY -c "print($left / 3600)")
  echo "== training (attempt $((attempt + 1)), $hours h left) $(date +%H:%M:%S)"
  "${LAUNCH[@]}" -m synth_rl.grpo --model "$MODEL" --out "$RUN/train" --steps "$STEPS" --max-hours "$hours" "$@" && break
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 20 ]; then echo "training failed $attempt times; giving up"; break; fi
  echo "training exited with an error; restarting from the last checkpoint in 30 s"
  sleep 30
done

ADAPTER=$RUN/train/final
[ -d "$ADAPTER" ] || ADAPTER=$RUN/train/ckpt/latest
bench() {  # KIND NAME AGENT [bench flags]: one benchmark job, results in eval_KIND/parts/NAME.jsonl
  local kind=$1 name=$2 out=$RUN/eval_$1
  shift 2
  $PY -m synth_rl.cli bench --kind "$kind" --specs mono --n "$EVAL_N" --seed 7 --out "$out" \
    --results "$out/parts/$name.jsonl" ${BLIND[@]+"${BLIND[@]}"} --agents "$@" > "$out/logs/$name.log" 2>&1 \
    || echo "benchmark $kind/$name failed; see $out/logs/$name.log"
}
if [ -d "$ADAPTER" ]; then
  echo "== benchmark: base model and adapter, with and without audio, on $GPUS GPU(s) $(date +%H:%M:%S)"
  KINDS="match repair edit"  # longest first, for a better spread over GPUs
  jobs=()
  for kind in $KINDS; do
    rm -rf "$RUN/eval_$kind/parts"
    mkdir -p "$RUN/eval_$kind/parts" "$RUN/eval_$kind/logs"
    bench "$kind" rule rule  # also writes the shared task file before the parallel jobs start
    jobs+=("$kind base 0" "$kind adapter 0" "$kind base-mute 1" "$kind adapter-mute 1")
  done
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
  else
    devices=()
    for ((d = 0; d < GPUS; d++)); do devices+=("$d"); done
  fi
  # Each GPU runs every n-th job in turn; one model per process, so no GPU ever holds two.
  for ((d = 0; d < ${#devices[@]}; d++)); do
    (
      for ((k = d; k < ${#jobs[@]}; k += ${#devices[@]})); do
        read -r kind name mute <<< "${jobs[$k]}"
        agent=$MODEL
        [[ $name == adapter* ]] && agent=$MODEL::$ADAPTER
        flags=()
        [ "$mute" = 1 ] && flags=(--mute-audio)
        CUDA_VISIBLE_DEVICES=${devices[$d]} bench "$kind" "$name" "$agent" ${flags[@]+"${flags[@]}"}
      done
    ) &
  done
  wait
  for kind in $KINDS; do
    echo "-- $kind"
    $PY -m synth_rl.cli summary --out "$RUN/eval_$kind"
  done
fi

echo "== packaging $(date +%H:%M:%S)"
items=()
for f in train/log.jsonl train/samples.jsonl train/config.json train/status.json train/final train/best \
         eval_edit eval_repair eval_match preflight.log run.log; do
  [ -e "$RUN/$f" ] && items+=("$f")
done
tar czf "$RUN/results.tar.gz" -C "$RUN" "${items[@]}"
echo "done: $RUN/results.tar.gz"
