"""One reply against the step loop, with and without audio, on the same tasks (the table in the README).

    python scripts/loop_vs_one_reply.py run    [--n 40] [--steps 6] [--workers 8] [--model qwen/qwen3.8-omni-flash]
    python scripts/loop_vs_one_reply.py report

Needs OPENROUTER_API_KEY. `run` draws --n edit and --n repair tasks from --seed, runs every task under four
conditions (oneshot, oneshot+mute, loop, loop+mute; a muted agent gets the same messages with silent clips), and
appends one row per episode to <out>/results.jsonl; rerunning continues where it stopped. Every row keeps the
edits of each step, so any other score can be computed later by replaying them. `report` prints means, paired
differences with bootstrap intervals over tasks, and how the loop episodes ended.
"""
import argparse
import json
import random
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

CONDITIONS = ("oneshot", "oneshot+mute", "loop", "loop+mute")
_w = {}


def _init(out: str, steps: int, model: str):
    from synth_rl.agents import OpenAIChatAgent
    from synth_rl.cli import load_tasks
    from synth_rl.vital_env import VitalEnv

    _w.update(env=VitalEnv(), tasks=load_tasks(Path(out) / "tasks.jsonl"), steps=steps,
              agents={False: OpenAIChatAgent(model), True: OpenAIChatAgent(model, mute_audio=True)})


def _param_scores(env, task, edits_per_step: list[dict]) -> list[dict]:
    """Repair only, after each step: how much of the gap to the target preset is closed, counted in parameters.
    restored: mean over the parameters that differ from the target of 1 - |final - target| / |start - target|
    (1 = set back exactly, 0 = untouched, negative = moved away; floored at -1). disturbed: parameters that were
    already right and were moved. error: sum |final - target| over all parameters / the same sum at the start."""
    from synth_rl.vital_env import moved

    env.load_state(task.target_state)
    target = env.values()
    env.load_state(task.state)
    start = env.values()
    wrong = moved(start, target, 1e-3)
    out = []
    for edits in edits_per_step:
        env.apply(edits)
        now = env.values()
        restored = [max(-1.0, 1 - abs(now[n] - target[n]) / abs(start[n] - target[n])) for n in wrong]
        out.append({"restored": float(np.mean(restored)), "n_wrong": len(wrong),
                    "disturbed": sum(1 for n in moved(start, now, 1e-3) if n not in wrong),
                    "error": sum(abs(now[n] - target[n]) for n in now) / sum(abs(start[n] - target[n]) for n in wrong)})
    return out


def _job(job):
    from synth_rl.cli import run_episode
    from synth_rl.loop import rollout

    condition, index = job
    env, task, agent = _w["env"], _w["tasks"][index], _w["agents"][condition.endswith("+mute")]
    error = ""
    for _ in range(4):
        try:
            if condition.startswith("oneshot"):
                result = run_episode(env, agent, task)
                edits = [result["turns"][0]["edits"]]
                row = {k: v for k, v in result.items() if k != "turns"}
                row.update(steps=[{"reward": result["reward"], "edits": edits[0], "reply": result["turns"][0]["raw"]}],
                           stopped="one shot", n_steps=1)
            else:
                result = rollout(agent, env, task, max_steps=_w["steps"])
                edits = [s["edits"] for s in result["steps"]]
                row = {k: v for k, v in result.items() if k not in ("turns", "messages")}
            break
        except RuntimeError as e:  # API error or a cut-off reply: try the episode again
            error = str(e)[:200]
    else:
        return {"condition": condition, "task": index, "kind": task.kind, "error": error}
    if task.kind == "repair":
        for step, scores in zip(row["steps"], _param_scores(env, task, edits)):
            step.update(scores)
    return {"condition": condition, "task": index, "kind": task.kind, "spec": task.spec_id, **row}


def run(args):
    from synth_rl.cli import save_tasks
    from synth_rl.procedural import random_patch
    from synth_rl.tasks import make_edit_task, make_repair_task
    from synth_rl.vital_env import VitalEnv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "tasks.jsonl").exists():
        env, rng, tasks = VitalEnv(), random.Random(args.seed), []
        for make in (make_edit_task, make_repair_task):
            made = 0
            while made < args.n:
                task = make(env, random_patch(env, rng), rng)
                if task:
                    tasks.append(task)
                    made += 1
        save_tasks(tasks, out / "tasks.jsonl")
    n_tasks = len((out / "tasks.jsonl").read_text().splitlines())
    results = out / "results.jsonl"
    seen = {(r["condition"], r["task"]) for r in map(json.loads, results.read_text().splitlines())} if results.exists() else set()
    jobs = [(c, i) for i in range(n_tasks) for c in CONDITIONS if (c, i) not in seen]
    print(f"{len(jobs)} episodes to run ({len(seen)} done)", flush=True)
    with get_context("spawn").Pool(args.workers, _init, (str(out), args.steps, args.model)) as pool, results.open("a") as f:
        for n, row in enumerate(pool.imap_unordered(_job, jobs), 1):
            if "error" in row:
                print("failed:", row, flush=True)
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"{n}/{len(jobs)}", flush=True)


def _ci(values, rng) -> str:
    """Mean with a 95 % bootstrap interval over tasks."""
    v = np.asarray(values, dtype=float)
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(4000)]
    low, high = np.percentile(boots, [2.5, 97.5])
    return f"{v.mean():+.3f} [{low:+.3f}, {high:+.3f}]"


def report(args):
    rows = [json.loads(line) for line in (Path(args.out) / "results.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(0)
    for kind in ("edit", "repair"):
        by = {c: {r["task"]: r for r in rows if r["kind"] == kind and r["condition"] == c} for c in CONDITIONS}
        tasks = sorted(set.intersection(*(set(v) for v in by.values())))
        print(f"\n=== {kind}: {len(tasks)} tasks with all four conditions ===")
        measures = ["reward"] + (["restored", "disturbed", "error"] if kind == "repair" else ["target", "drift", "n_changed"])

        def value(row, measure, step=-1):
            s = row["steps"][step]
            return s[measure] if measure in s else row[measure]

        for measure in measures:
            print(f"\n{measure}: mean [95 % interval]")
            for c in CONDITIONS:
                final = _ci([value(by[c][t], measure) for t in tasks], rng)
                if c.startswith("loop") and measure in by[c][tasks[0]]["steps"][0]:
                    first = _ci([value(by[c][t], measure, 0) for t in tasks], rng)
                    print(f"  {c:13s} after step 1 {first}   final {final}")
                else:
                    print(f"  {c:13s} {final}")
            if measure in ("n_changed", "target", "drift"):
                continue
            pairs = [("loop final - oneshot", lambda t: value(by["loop"][t], measure) - value(by["oneshot"][t], measure)),
                     ("loop step 1 - oneshot", lambda t: value(by["loop"][t], measure, 0) - value(by["oneshot"][t], measure)),
                     ("loop final - loop step 1", lambda t: value(by["loop"][t], measure) - value(by["loop"][t], measure, 0)),
                     ("oneshot - oneshot+mute", lambda t: value(by["oneshot"][t], measure) - value(by["oneshot+mute"][t], measure)),
                     ("loop - loop+mute", lambda t: value(by["loop"][t], measure) - value(by["loop+mute"][t], measure))]
            for name, diff in pairs:
                print(f"  paired: {name:26s} {_ci([diff(t) for t in tasks], rng)}")
        for c in ("loop", "loop+mute"):
            eps = [by[c][t] for t in tasks]
            steps = [e["n_steps"] for e in eps]
            stops = {s: sum(e["stopped"] == s for e in eps) for s in sorted({e["stopped"] for e in eps})}
            rewards = [[s["reward"] for s in e["steps"]] for e in eps]
            worse = sum(r[-1] < r[0] - 1e-9 for r in rewards)
            better = sum(r[-1] > r[0] + 1e-9 for r in rewards)
            best = np.mean([max(r) for r in rewards])
            down = sum(b < a - 1e-9 for r in rewards for a, b in zip(r, r[1:]))
            up = sum(b > a + 1e-9 for r in rewards for a, b in zip(r, r[1:]))
            print(f"\n{c}: steps per episode mean {np.mean(steps):.2f} (incl. the closing step), stops {stops}")
            print(f"  final vs step 1: better in {better}, worse in {worse}, same in {len(eps) - better - worse}; "
                  f"mean of the best step {best:+.3f}; later steps that raised the reward {up}, lowered it {down}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--model", default="qwen/qwen3.8-omni-flash", help="OpenRouter model id (one with audio input)")
    ap.add_argument("--out", default="outputs/agent_loop/measure")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    (run if args.cmd == "run" else report)(args)
