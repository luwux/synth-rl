# synth-rl

Audio-LLM agents that program a synthesizer. Describe a change in words ("brighter, with a shorter tail") or hand over a recording, and an agent listens, turns the knobs of [Vital](https://vital.audio), listens to its own result, and writes out a preset you can open in Vital.

Current models are not good at this, especially when the answer is only in the audio. So the repository also holds the environment to measure that: a benchmark scored on the sound each agent actually produces, and a GRPO recipe that fine-tunes Qwen Omni models with LoRA against that reward.

Vital runs headless through the [`vita`](https://github.com/DBraun/Vita) Python bindings, which ship Linux, macOS, and Windows wheels. Rendering a 3-second note takes about 0.05 s on one CPU core.

## Use it

```sh
uv venv --python 3.12 && uv pip install -e .              # add ".[train]" for local models and training
export GEMINI_API_KEY=...

synth-rl edit --preset my.vital --instruction "brighter, with a shorter tail"
synth-rl match --target note.wav                           # one note at middle C, up to 3 s
```

- **`edit`** changes a preset to follow an instruction. The agent hears the sound, edits, hears its own result, and refines it (two rounds by default).
- **`match`** builds a patch that sounds like a recording, starting from Vital's init patch (or from `--preset`).

Both print each round's changes and reasons, and write the result to `--out` (default `outputs/edit` or `outputs/match`):
- `final.vital`: the edited preset, ready to open in Vital;
- `round0_before.wav` and `final.wav`: the sound before and after;
- `result.json`: every prompt, reply, and change.

`--agent` takes a Gemini model id (the default, `gemini-3.8-flash`), a local Qwen Omni model as a Hugging Face id or folder (`Qwen/Qwen2.5-Omni-7B`, `Qwen/Qwen3-Omni-30B-A3B-Instruct`), a fine-tuned LoRA as `<model>::<adapter folder>`, or `rule`, a keyword baseline that never listens.

## How it works

The benchmark and the training run on generated tasks:

1. **Starting sound.** By default each task starts from a procedurally generated patch (random oscillators, filter, envelopes, envelope-to-cutoff modulation, and effects, built from Vital's init preset), so no third-party presets are needed and renders are bit-exact. Any folder of `.vital` presets works too.
2. **Task.** There are three kinds:
   - **edit:** the patch and an instruction it can still satisfy (a sound that is already fully dry gets no "make it drier").
   - **repair:** 2–4 parameters of the patch are randomized; the agent hears the broken sound and the original and must restore it.
   - **match:** the agent starts from Vital's init patch, hears the target, and must rebuild it from scratch. Every procedural target is reachable exactly through the editable controls (a preflight check rebuilds targets from their values).
3. **Observation.** The agent sees 36 curated controls (oscillators, filter, envelopes, the envelope-to-cutoff amount, effects) with their normalized 0–1 values and Vital's display text, the instruction, and the rendered audio (plus the target for repair and match). In the **blind** variant the current values are hidden, so the agent has to infer them by listening.
4. **Action.** JSON edits: `{"edits": {"filter_1_cutoff": 0.8}, "reason": "..."}`.
5. **Reward.** The environment applies the edits, renders, and computes timbre descriptors: spectral centroid, high-frequency energy, peak level, attack time, sustain ratio, release tail, and stereo width.
   - **Edit reward:** the `tanh`-shaped signed change of the descriptors the instruction targets, minus drift in descriptors it should leave alone, minus level drift beyond 3 dB, minus a small penalty for touching many parameters. Leaving the sound unchanged scores exactly 0.
   - **Repair and match reward:** the relative improvement in multi-resolution STFT distance to the target, `tanh`-shaped.

Agents may run several rounds, hearing their own render each round.

## Benchmark

```sh
synth-rl bench --agents rule --n 30                        # keyword baseline that never listens
synth-rl bench --agents gemini-3.8-flash --n 30
synth-rl bench --agents Qwen/Qwen2.5-Omni-7B --n 30        # local Qwen Omni (Hugging Face id or folder)
synth-rl bench --agents Qwen/Qwen2.5-Omni-7B --mute-audio  # same prompts with silent audio: does it listen?
synth-rl bench --agents Qwen/Qwen2.5-Omni-7B --blind       # hidden parameter values: it has to listen
synth-rl bench --kind repair --agents ...                  # restore a broken sound
synth-rl bench --kind match --agents ...                   # rebuild a target sound from the init patch
synth-rl demo --agent gemini-3.8-flash --rounds 2          # full trajectories: prompt, reply, changes, audio
```

`bench` saves its task set to `<out>/tasks.jsonl` and appends to `<out>/results.jsonl`, so agents can be run one at a time (large local models never share memory) and compared on identical tasks. Parallel runs write separate files with `--results <out>/parts/<name>.jsonl`; `synth-rl summary --out <out>` merges them. `--presets ~/Music/Vital` switches from procedural patches to a preset folder; scores then average three renders per state, because presets with free-running LFOs or random modulators do not render identically. `synth-rl rescore --out <dir>` recomputes rewards from stored edits after a reward change. Local Omni models hear 16 kHz mono, so `--specs mono` (the default for training) drops the stereo-width instructions.

## Training

`synth_rl/grpo.py` trains a LoRA on the language model of a Qwen2.5-Omni or Qwen3-Omni thinker with GRPO:

- Each step samples fresh procedural tasks (edit, repair, and match in equal shares by default; `--kinds` selects), draws `--group` completions per task at temperature 1, and scores every completion by rendering it. Replies may run to `--max-new-tokens` 512, enough to set every control for a match.
- Advantages are the group-normalized rewards. Groups whose completions all score the same carry no signal and are skipped.
- One gradient step is taken per batch, so the update is on-policy. An optional KL penalty to the base model is available (`--kl`).
- Unparseable replies score −3, below every real attempt (real edit rewards are bounded below by about −2.4).
- Evaluation on a fixed task set runs every `--eval-every` steps, both with the real audio and with silence, and reports each task kind separately. The gap between the two shows how much the policy relies on listening. The first evaluation, before any update, is the base model.
- Checkpoints are atomic. Rerunning the same command resumes.

```sh
python -m synth_rl.grpo --model models/Qwen2.5-Omni-7B --out runs/edit-7b              # one GPU (or MPS/CPU)
torchrun --nproc_per_node 8 -m synth_rl.grpo --model models/Qwen2.5-Omni-7B --out runs/edit-7b
python -m synth_rl.grpo ... --blind                                                     # hide parameter values too
```

Outputs:

- `log.jsonl`: per-step reward (overall and per task kind), validity, skipped groups, completion length, loss, gradient norm, and timings; evaluations.
- `samples.jsonl`: example completions.
- `best/` and `final/`: LoRA adapters.

Benchmark an adapter with `--agents <model>::<adapter dir>`.

**One command on a fresh GPU machine.** `scripts/run_all.sh` runs everything in order:

1. Installs the pinned environment with uv.
2. Downloads the model from Hugging Face.
3. Runs preflight checks (`synth_rl/preflight.py`) and stops on any failure:
   - rendering is deterministic;
   - reward edge cases score correctly;
   - reward worker processes agree with the main process;
   - the model's replies parse;
   - two training steps on every GPU, a resume, and a benchmark of the saved adapter.
4. Starts training in the background. On errors it restarts from the last checkpoint until the time budget is used.
5. Benchmarks the keyword baseline, the base model, and the adapter on fixed edit, repair, and match sets, with and without audio, spread over the GPUs, and packs logs, adapters, and benchmark results into `runs/<name>/results.tar.gz`. The last hour of the budget is kept for this (`EVAL_RESERVE`, in seconds).

```sh
scripts/run_all.sh --model Qwen/Qwen2.5-Omni-7B --hours 20
scripts/run_all.sh --model Qwen/Qwen2.5-Omni-3B --gpus 1 --hours 4 -- --blind
```

Behind a restricted network, point the downloads at mirrors first, for example `export HF_ENDPOINT=https://hf-mirror.com UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`, or pass a local model folder as `--model`.

Estimated memory per GPU (not yet measured on CUDA):

- **Qwen2.5-Omni-7B (bf16, LoRA, gradient checkpointing):** about 40 GB with the default batch sizes.
- **Qwen2.5-Omni-3B:** about 9 GB of weights; 24 GB cards should work with smaller batches, such as `--gen-batch 16 --micro-batch 2`.
- **Qwen3-Omni-30B-A3B:** holds the full 60 GB model on each GPU. The fused MoE experts take no LoRA, so only attention layers are adapted. Lower `--gen-batch` and `--micro-batch` on 80 GB cards.

### Other RL frameworks

`synth_rl/rl.py` exposes the reward with the signatures these frameworks expect:

- `compute_score`: verl's custom-reward signature.
- `SynthEditORM`: ms-swift's outcome-reward plugin interface (not yet tested against ms-swift).

`synth-rl export` writes single-turn prompts with rendered audio for frameworks that read datasets from disk. Each row carries the starting state, so reward workers re-render from scratch.

## Status

Early. Known limitations:

- Generation uses Hugging Face `generate`. A vLLM rollout backend would make sampling several times faster, but audio-input support for these models in vLLM still has to be checked.
- Descriptor thresholds and reward weights are hand-set. The edit reward rewards bigger moves in the requested direction until `tanh` saturates.
- When parameter values are shown, edit tasks can largely be solved from text alone. Use the blind variant, repair and match tasks, and the muted-audio comparison to measure listening.
- Creating a sound from a text description alone ("a warm, slowly evolving pad") is not covered yet. Match tasks cover creation from a reference sound.

## License

GPL-3.0-or-later, following Vital.
