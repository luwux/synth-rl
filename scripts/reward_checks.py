"""Do the rewards order actions of known quality the way they should? No model is called.

    python scripts/reward_checks.py [--out outputs/agent_loop/measure] [--n 40] [--seed 20261116]

Uses the task set of `loop_vs_one_reply.py` (<out>/tasks.jsonl, drawn here from --seed if it is not there yet).

Repair. The target preset is known, so the quality of an action is known by construction:
- every wrong parameter moved a fraction of the way to its target value (-0.5 = away from it, 1 = the full repair);
- the full repair written as an agent writes it (text, three significant digits);
- one wrong parameter set right, one right parameter moved by 0.25 of its range, and both together;
- the target patch with a single parameter 0.0004 of its range off (the reward's sensitivity to tiny moves).
Edit. The keyword rule, which reads only the request, under the edit reward: what a score needs no listening for.
If <out>/results.jsonl exists, also how the spectral reward and the parameter score agree on a model's episodes.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth_rl.agents import RuleAgent, parse_edits, reply_text  # noqa: E402
from synth_rl.cli import load_tasks, save_tasks  # noqa: E402
from synth_rl.procedural import random_patch  # noqa: E402
from synth_rl.tasks import make_edit_task, make_repair_task, mss_distance, repair_reward, score  # noqa: E402
from synth_rl.vital_env import EDITABLE, VitalEnv, moved  # noqa: E402

FRACTIONS = (-0.5, 0.25, 0.5, 0.75, 1.0)
TINY = 0.0004


def tasks_of(out: Path, n: int, seed: int, env: VitalEnv) -> list:
    if not (out / "tasks.jsonl").exists():  # the same draw as loop_vs_one_reply.py
        rng, tasks = random.Random(seed), []
        for make in (make_edit_task, make_repair_task):
            made = 0
            while made < n:
                task = make(env, random_patch(env, rng), rng)
                if task:
                    tasks.append(task)
                    made += 1
        out.mkdir(parents=True, exist_ok=True)
        save_tasks(tasks, out / "tasks.jsonl")
    return load_tasks(out / "tasks.jsonl")


def repair_checks(env: VitalEnv, tasks: list) -> None:
    rng = random.Random(0)
    path, written, single, other, both, starts, tiny = [], [], [], [], [], [], {}
    for task in tasks:
        env.load_state(task.target_state)
        target, target_audio = env.values(), env.render(task.target_state)
        env.load_state(task.state)
        start, params = env.values(), env.params()
        wrong = moved(start, target, 1e-3)
        before = mss_distance(env.render(task.state), target_audio)
        starts.append(before)

        def reward(edits: dict) -> float:
            env.load_state(task.state)
            env.apply(edits)
            return repair_reward(before, mss_distance(env.render(env.state), target_audio))["reward"]

        path.append([reward({n: float(np.clip(start[n] + f * (target[n] - start[n]), 0, 1)) for n in wrong})
                     for f in FRACTIONS])
        fix = {n: target[n] for n in wrong}
        written.append(score(env, task, parse_edits(reply_text(fix), params))["reward"])
        single += [reward({n: target[n]}) for n in wrong]
        name = rng.choice([n for n in start if n not in wrong and not n.endswith("_on")])
        nudge = {name: start[name] + (0.25 if start[name] < 0.5 else -0.25)}
        other.append(reward(nudge))
        both.append(reward({**fix, **nudge}))
        for name in EDITABLE:
            if name in target and not name.endswith("_on"):
                env.load_state(task.target_state)
                env.apply({name: target[name] + (TINY if target[name] < 0.5 else -TINY)})
                tiny.setdefault(name, []).append(mss_distance(env.render(env.state), target_audio))

    def signs(values) -> str:
        v = np.asarray(values)
        return f"mean {v.mean():+.2f}; above 0 in {int((v > 0.02).sum())}, at 0 in {int((abs(v) <= 0.02).sum())}, " \
               f"below 0 in {int((v < -0.02).sum())} of {len(v)}"

    path = np.array(path)
    print(f"repair, {len(tasks)} tasks (leaving the sound alone scores 0)")
    for f, column in zip(FRACTIONS, path.T):
        print(f"  every wrong parameter moved {f:+.2f} of the way to its target: {signs(column)}")
    steps = np.diff(np.c_[np.zeros(len(path)), path[:, 1:]], axis=1)
    print(f"  the reward never falls from 0 through 0.25, 0.5, 0.75 to 1: {int((steps > -0.02).all(1).sum())} of {len(path)} tasks")
    print(f"  full repair written as text: mean {np.mean(written):.3f}, lowest {min(written):.3f}")
    print(f"  one wrong parameter set right: {signs(single)}")
    print(f"  one right parameter moved by 0.25 of its range: {signs(other)}")
    print(f"  full repair plus that move: mean {np.mean(both):+.2f}")
    print(f"  spectral distance at the start of a task: median {np.median(starts):.2f}, lowest {min(starts):.2f}")
    print(f"  the same distance after moving one parameter of the target patch by {TINY} of its range")
    for name, d in sorted(tiny.items(), key=lambda item: -max(item[1])):
        if max(d) > 0.1:
            print(f"    {name:22s} median {np.median(d):.2f}, highest {max(d):.2f}, above 0.1 in {int((np.array(d) > 0.1).sum())} of {len(d)}")


def edit_checks(env: VitalEnv, tasks: list) -> None:
    rule, by = RuleAgent(), {}
    for task in tasks:
        env.load_state(task.state)
        result = score(env, task, rule.propose(task.instruction, env.params(), None)["edits"])
        by.setdefault(task.spec_id, []).append(result["reward"])
    rewards = [r for v in by.values() for r in v]
    print(f"edit, {len(tasks)} tasks: the keyword rule (reads only the request) scores {np.mean(rewards):.3f}")
    print("  by request:", ", ".join(f"{k} {np.mean(v):.2f} ({len(v)})" for k, v in sorted(by.items())))


def agreement(results: Path) -> None:
    from scipy.stats import spearmanr

    rows = [json.loads(line) for line in results.read_text().splitlines()]
    print("repair, a model's episodes: spectral reward against the parameter score (1 - error left / error at the start)")
    for condition in sorted({r["condition"] for r in rows}):
        last = [r["steps"][-1] for r in rows if r["kind"] == "repair" and r["condition"] == condition]
        reward = np.array([s["reward"] for s in last])
        clean = np.array([s["disturbed"] == 0 for s in last])
        print(f"  {condition:13s} rank correlation {spearmanr(reward, [1 - s['error'] for s in last]).statistic:+.2f}; "
              f"reward below 0 in {int((reward < -0.02).sum())} of {len(last)} episodes, "
              f"{int(((reward < -0.02) & clean).sum())} of them without a right parameter moved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/agent_loop/measure")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261116)
    args = ap.parse_args()
    env = VitalEnv()
    tasks = tasks_of(Path(args.out), args.n, args.seed, env)
    repair_checks(env, [t for t in tasks if t.kind == "repair"])
    edit_checks(env, [t for t in tasks if t.kind == "edit"])
    if (Path(args.out) / "results.jsonl").exists():
        agreement(Path(args.out) / "results.jsonl")
