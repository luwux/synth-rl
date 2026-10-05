"""Preflight checks before a long training run. Exits non-zero on the first failure.

1. render: procedural patches render, render identically twice, and can be rebuilt exactly from the init patch
   through the editable parameters (so every match target is reachable)
2. reward: no edit scores exactly 0, unparseable output scores FORMAT_FAIL, the keyword baseline scores
   clearly above 0, and reward worker processes agree with the main process
3. model: the policy loads and its greedy replies parse as edits
4. training: two GRPO steps (on every GPU when --gpus > 1), checkpoint, resume for a third step,
   and benchmark the saved LoRA adapter
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .agents import RuleAgent
from .procedural import random_patch
from .rl import FORMAT_FAIL, reward_from_row
from .tasks import MONO_SPECS, make_edit_task, make_match_task, score
from .vital_env import VitalEnv

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        raise SystemExit(1)


def _row_reward(item):
    return reward_from_row(*item)


def check_env() -> list:
    env = VitalEnv()
    rng = random.Random(0)
    t0 = time.time()
    tasks = []
    while len(tasks) < 8:
        task = make_edit_task(env, random_patch(env, rng), rng, MONO_SPECS)
        if task:
            tasks.append(task)
    check("render: procedural tasks", True, f"8 tasks in {time.time() - t0:.1f}s")
    diffs = [float(np.abs(env.render(t.state) - env.render(t.state)).max()) for t in tasks]
    check("render: deterministic", max(diffs) == 0.0, f"max diff over 8 patches {max(diffs):.2e}")
    gaps = []
    for t in tasks:
        env.load_state(t.state)
        exact = env.values()
        match = make_match_task(env, t.state)
        env.load_state(match.state)
        env.apply(exact)
        gaps.append(float(np.abs(env.render(env.state) - env.render(t.state)).max()))
    check("render: targets rebuilt from the init patch", max(gaps) < 1e-4, f"max diff {max(gaps):.2e}")

    check("reward: no edit is 0", score(env, tasks[0], {})["reward"] == 0.0)
    row = {"kind": "edit", "state": tasks[0].state, "instruction": tasks[0].instruction, "spec_id": tasks[0].spec_id}
    check("reward: unparseable is FORMAT_FAIL", reward_from_row(row, "I would raise the cutoff.") == FORMAT_FAIL)
    rule = RuleAgent()
    rule_rewards = []
    for t in tasks:
        env.load_state(t.state)
        rule_rewards.append(score(env, t, rule.propose(t.instruction, env.params(), None)["edits"])["reward"])
    check("reward: keyword baseline > 0.2", float(np.mean(rule_rewards)) > 0.2, f"mean {np.mean(rule_rewards):.3f}")

    items = []
    for t in tasks[:4]:
        env.load_state(t.state)
        edits = rule.propose(t.instruction, env.params(), None)["edits"]
        items.append(({"kind": "edit", "state": t.state, "instruction": t.instruction, "spec_id": t.spec_id},
                      json.dumps({"edits": edits})))
    with mp.get_context("spawn").Pool(2) as pool:
        remote = pool.map(_row_reward, items)
    local = [reward_from_row(*it) for it in items]
    check("reward: worker processes agree", np.allclose(remote, local), f"{np.round(remote, 3).tolist()}")
    return tasks


def check_model(model: str, tasks: list) -> None:
    from .omni import OmniAgent

    t0 = time.time()
    agent = OmniAgent(model)
    check("model: load", True, f"{agent.policy.family} on {agent.policy.device} in {time.time() - t0:.0f}s")
    env = VitalEnv()
    parsed = 0
    for t in tasks[:4]:
        env.load_state(t.state)
        out = agent.propose(t.instruction, env.params(), env.render(t.state))
        parsed += bool(out["edits"])
    check("model: replies parse as edits", parsed >= 2, f"{parsed}/4 parsed; last reply: {out['raw'][:120]!r}")
    del agent
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def check_training(model: str, out: Path, gpus: int) -> None:
    shutil.rmtree(out, ignore_errors=True)
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}"] \
        if gpus > 1 else [sys.executable]
    base = launcher + ["-m", "synth_rl.grpo", "--model", model, "--out", str(out), "--prompts", "2", "--group", "4",
                       "--gen-batch", "8", "--micro-batch", "2", "--eval-every", "2", "--eval-n", str(4 * gpus),
                       "--eval-batch", "4", "--save-every", "1", "--reward-workers", "2", "--max-new-tokens", "256"]
    for steps in (2, 3):
        t0 = time.time()
        proc = subprocess.run(base + ["--steps", str(steps)], capture_output=True, text=True)
        log = out / "train.log"
        with log.open("a") as f:
            f.write(proc.stdout + proc.stderr)
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        check(f"training: run to step {steps}", proc.returncode == 0, f"{time.time() - t0:.0f}s" if proc.returncode == 0
              else f"exit {proc.returncode}, see {log}\n{tail}")
    steps = [json.loads(line)["step"] for line in (out / "log.jsonl").read_text().splitlines()
             if "eval" not in json.loads(line)]
    check("training: resumed without repeating steps", steps == [1, 2, 3], f"logged steps {steps}")
    check("training: adapter saved", (out / "final" / "adapter_model.safetensors").exists())

    bench_out = out / "bench"
    proc = subprocess.run([sys.executable, "-m", "synth_rl.cli", "bench", "--agents", f"{model}::{out / 'final'}",
                           "--specs", "mono", "--n", "2", "--out", str(bench_out)], capture_output=True, text=True)
    ok = proc.returncode == 0 and (bench_out / "summary.json").exists()
    check("training: fine-tuned adapter benchmarks", ok, "" if ok else (proc.stdout + proc.stderr)[-1500:])


def main() -> None:
    p = argparse.ArgumentParser(prog="synth-rl-preflight", description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="Qwen Omni folder")
    p.add_argument("--out", default="outputs/preflight")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--skip-training", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    tasks = check_env()
    check_model(args.model, tasks)
    if not args.skip_training:
        check_training(args.model, Path(args.out), args.gpus)
    print(f"preflight passed: {len(results)} checks in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
