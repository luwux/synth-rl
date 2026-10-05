# Notes for coding agents

## Running a training job

Use the one command; do not assemble the steps by hand:

```sh
scripts/run_all.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct --hours <hours available>
```

It installs the environment, downloads the model, runs the preflight checks (`synth_rl/preflight.py`), trains in the background with automatic restarts, benchmarks before and after, and writes `runs/<name>/results.tar.gz`.

- Start it inside `tmux` or `screen`: the preflight runs in the foreground and dies with the SSH session. Training then continues in the background.
- On Slurm, run `scripts/run_all.sh ... --foreground` inside the job; background processes are killed when the allocation ends.
- To split one machine into two runs, give each its own cards: `CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/run_all.sh ...`.
- After a crash or reboot, resume with the `scripts/train.sh ...` command that `run_all.sh` printed; it continues from `runs/<name>/train/ckpt/latest`.
- Memory per GPU (estimates; not yet measured on CUDA): Qwen3-Omni-30B-A3B needs 80 GB cards; Qwen2.5-Omni-7B about 40 GB; Qwen2.5-Omni-3B fits 24 GB with smaller batches. Disk: about 100 GB for the 30B model (66 GB of weights plus the CUDA environment and checkpoints), about 35 GB for 7B.

## Where to look

| File in `runs/<name>/` | Contents |
| --- | --- |
| `preflight.log` | `PASS`/`FAIL` per check; the run stops at the first `FAIL` |
| `run.log` | training attempts, restarts, final benchmark |
| `train/log.jsonl` | one line per step (`reward`, `valid`, `skipped_groups`, `secs`, `by_kind`) and per evaluation (`"eval"` lines) |
| `train/samples.jsonl` | example completions |
| `train/status.json` | last step and best evaluation |

A healthy run has `valid` near 1, `skipped_groups` well below 0.5, and evaluation reward rising over the first 50–100 steps.

## Diagnosing

`scripts/doctor.sh [runs/<name>]` writes `doctor.txt` with GPUs, driver, PyTorch, disk, network, and the ends of every log. Start there.

| Symptom | Likely cause and fix |
| --- | --- |
| Downloads hang or fail (uv, Python, PyPI, Hugging Face) | Restricted network. Set mirrors before running: `HF_ENDPOINT`, `UV_DEFAULT_INDEX`, `PIP_INDEX_URL` (see README). Or download the model separately and pass its folder as `--model`. |
| "nvidia-smi sees GPUs but PyTorch cannot use them" | The NVIDIA driver is older than this PyTorch build needs. Update the driver, or install the same torch version built for an older CUDA from the PyTorch package index into `.venv`. |
| `out of memory, retrying with ...` in the log | Expected: training halves `--gen-batch` or `--micro-batch` and retries. Only a failure at batch size 1 means the model does not fit; use larger cards or a smaller model, or lower `--max-new-tokens`. |
| Multi-GPU start hangs or fails in NCCL | Check that `--gpus 1` works. Then try `NCCL_P2P_DISABLE=1` or `NCCL_IB_DISABLE=1`. |
| Preflight "replies parse as edits" fails | Incomplete or wrong model download; check `models/<name>`. |
| Repeated restarts in `run.log` | Read the first traceback; the restart loop gives up after 20 failures. |

Do not change rewards, task generation, or check thresholds to make a check pass, and do not delete `train/ckpt/`, the resume point. Report the failing check and `doctor.txt` instead.

## Code map

- `synth_rl/vital_env.py`: headless Vital, the 36 editable parameters, rendering.
- `synth_rl/procedural.py`, `synth_rl/tasks.py`: generated patches, the edit, repair, and match tasks, and their rewards.
- `synth_rl/agents.py`, `synth_rl/omni.py`: prompts, reply parsing, Gemini and local Qwen Omni agents.
- `synth_rl/grpo.py`: GRPO with LoRA, single or multi GPU (torchrun).
- `synth_rl/cli.py`: `edit`, `match`, `bench`, and the other commands.
- `synth_rl/preflight.py`: the checks run before training; also the closest thing to a test suite.
