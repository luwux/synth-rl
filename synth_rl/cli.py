"""Command line: run one edit, benchmark agents, measure render noise, export RL prompts."""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .agents import SYSTEM, GeminiAgent, RuleAgent, hide_values, prompt_text
from .features import describe
from .procedural import init_patch, random_patch
from .tasks import (EDIT_SPECS, MATCH_INSTRUCTION, MONO_SPECS, REPAIR_INSTRUCTION, Task, changed_params,
                    make_edit_task, make_match_task, make_repair_task, score_against_audio, score_state)
from .vital_env import SR, VitalEnv, list_presets, load_audio, preset_name

DEFAULT_PRESETS = "procedural"


def make_agent(name: str, mute_audio: bool):
    """'rule', a Qwen Omni folder or Hugging Face id (optionally '<model>::<lora_adapter>'), or a Gemini model id."""
    if name == "rule":
        return RuleAgent()
    path, _, adapter = name.partition("::")
    if not Path(path).expanduser().exists() and "omni" in path.lower() and "/" in path:
        from huggingface_hub import snapshot_download

        path = snapshot_download(path)
    if (Path(path).expanduser() / "config.json").exists():
        from .omni import OmniAgent

        return OmniAgent(str(Path(path).expanduser()), adapter or None, mute_audio=mute_audio)
    return GeminiAgent(model=name, mute_audio=mute_audio)


def starting_states(env: VitalEnv, presets: str, rng: random.Random):
    """Endless starting states: procedural patches, or random presets from a folder of .vital files."""
    files = None if presets == "procedural" else list_presets(presets)
    if files is not None and not files:
        raise SystemExit(f"no .vital presets under {presets}")
    while True:
        yield random_patch(env, rng) if files is None else env.load_preset(rng.choice(files))


def make_tasks(env: VitalEnv, args, rng: random.Random) -> list[Task]:
    specs = MONO_SPECS if args.specs == ["mono"] else args.specs
    tasks = []
    for state in starting_states(env, args.presets, rng):
        if args.kind == "edit":
            task = make_edit_task(env, state, rng, specs)
        elif args.kind == "repair":
            task = make_repair_task(env, state, rng)
        else:
            task = make_match_task(env, state)
        if task:
            tasks.append(task)
        if len(tasks) == args.n:
            return tasks


def save_tasks(tasks: list[Task], path: Path) -> None:
    with path.open("w") as f:
        for t in tasks:
            f.write(json.dumps({"kind": t.kind, "state": t.state, "instruction": t.instruction, "spec_id": t.spec_id,
                                "target_state": t.target_state, "perturbed": t.perturbed}) + "\n")


def load_tasks(path: Path) -> list[Task]:
    return [Task(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def summarize(results: Path) -> dict:
    """Mean reward per agent, overall and per instruction family, over every row in results.jsonl."""
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    out = {}
    for agent in dict.fromkeys(r["agent"] for r in rows):
        mine = [r for r in rows if r["agent"] == agent]
        rewards = np.array([r["reward"] for r in mine], dtype=float)
        by_spec = {}
        for r in mine:
            by_spec.setdefault(r["spec"] or r["kind"], []).append(r["reward"])
        out[agent] = {"mean": round(float(np.nanmean(rewards)), 3), "n": int(np.sum(~np.isnan(rewards))),
                      "no_change": sum(1 for r in mine if r.get("n_changed", 1) == 0),
                      "by_spec": {k: round(float(np.nanmean(v)), 3) for k, v in sorted(by_spec.items())}}
    return out


def run_episode(env: VitalEnv, agent, task: Task, rounds: int = 1, out_dir: Path | None = None,
                renders: int = 1, blind: bool = False, target_audio: np.ndarray | None = None) -> dict:
    """Let the agent edit for `rounds` turns, hearing the latest render each turn, then score.

    `target_audio` is a recorded target for match tasks without a target preset. Edit tasks without an
    instruction family (free-form instructions) are not scored."""
    state = task.state
    if task.target_state:
        target_audio = env.render(task.target_state)
    turns = []
    for r in range(rounds):
        env.load_state(state)
        params = env.params()
        audio = env.render(state)
        if out_dir:
            sf.write(out_dir / f"round{r}_before.wav", audio.T, SR)
        shown = hide_values(params) if blind else params
        proposal = agent.propose(task.instruction, shown, audio, target_audio)
        env.load_state(state)
        applied = env.apply(proposal["edits"])
        state = env.state
        new_params = env.params()
        changes = {k: {"from": params[k]["display"], "to": new_params[k]["display"],
                       "from_norm": params[k]["value"], "to_norm": new_params[k]["value"]} for k in applied}
        turns.append({"prompt": prompt_text(shown, task.instruction), "raw": proposal.get("raw", ""),
                      "edits": {k: proposal["edits"][k] for k in applied}, "changes": changes,
                      "reason": proposal.get("reason", "")})
    if task.kind == "edit" and not task.spec_id:
        result = {"n_changed": len(changed_params(env, task.state, state))}
    elif task.kind != "edit" and not task.target_state:
        result = score_against_audio(env, task.state, state, target_audio)
    else:
        result = score_state(env, task, state, renders)
    if out_dir:
        sf.write(out_dir / "final.wav", env.render(state).T, SR)
        if target_audio is not None:
            sf.write(out_dir / "target.wav", target_audio.T, SR)
        (out_dir / "final.vital").write_text(state)
    return {"turns": turns, **result}


def report(out: Path, result: dict) -> None:
    for n, turn in enumerate(result["turns"], 1):
        print(f"round {n}: {turn['reason']}")
        for name, c in turn["changes"].items():
            if c["from_norm"] == c["to_norm"]:
                continue
            print(f"  {name:22s} {c['from'] or c['from_norm']} -> {c['to'] or c['to_norm']}")
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"edited preset: {out / 'final.vital'}   audio: {out}/round0_before.wav, {out / 'final.wav'}")


def cmd_edit(args) -> None:
    """Edit a preset from a natural-language instruction; writes the edited .vital and before/after audio."""
    env = VitalEnv()
    state = env.load_preset(Path(args.preset).expanduser())
    task = Task("edit", state, args.instruction, spec_id=args.spec or "")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    before = describe(env.render(state))
    result = run_episode(env, make_agent(args.agent, args.mute_audio), task, args.rounds, out)
    result.update(preset=preset_name(state), before=before, after=describe(env.render(env.state)))
    report(out, result)
    print("timbre before -> after: " + ", ".join(f"{k} {result['before'][k]:.2f} -> {result['after'][k]:.2f}"
                                                 for k in result["before"]))
    if "reward" in result:
        print(f"reward ({args.spec}): {result['reward']:.3f}")


def cmd_match(args) -> None:
    """Build a patch that sounds like a recording, starting from the init patch or a given preset."""
    env = VitalEnv()
    target = load_audio(args.target)
    if args.preset:
        task = Task("repair", env.load_preset(Path(args.preset).expanduser()), REPAIR_INSTRUCTION)
    else:
        task = Task("match", init_patch(env), MATCH_INSTRUCTION)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = run_episode(env, make_agent(args.agent, args.mute_audio), task, args.rounds, out, target_audio=target)
    report(out, result)
    print(f"spectral distance to the target: {result['dist_before']:.3f} -> {result['dist_after']:.3f}"
          f" (reward {result['reward']:.3f})")


def cmd_bench(args) -> None:
    """Score agents on one shared task set. Agents run one at a time so large local models never coexist;
    rerunning with the same --out and other agents adds rows to the same results.jsonl."""
    env = VitalEnv()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task_file = Path(args.tasks) if args.tasks else out / "tasks.jsonl"
    if task_file.exists():
        tasks = load_tasks(task_file)[: args.n]
    else:
        tasks = make_tasks(env, args, random.Random(args.seed))
        save_tasks(tasks, task_file)
    renders = args.renders or (1 if args.presets == "procedural" else 3)
    results = Path(args.results) if args.results else out / "results.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    for name in args.agents:
        t_load = time.time()
        agent = make_agent(name, args.mute_audio)
        print(f"loaded {agent.name} in {time.time() - t_load:.1f}s")
        with results.open("a") as log:
            for i, task in enumerate(tasks):
                t0 = time.time()
                try:
                    res = run_episode(env, agent, task, args.rounds, renders=renders, blind=args.blind)
                except Exception as e:  # keep the benchmark running past API or generation errors
                    res = {"reward": float("nan"), "error": repr(e)}
                row = {"i": i, "agent": agent.name + ("+blind" if args.blind else ""), "rounds": args.rounds, "kind": task.kind, "spec": task.spec_id,
                       "instruction": task.instruction, "preset": preset_name(task.state),
                       "secs": round(time.time() - t0, 2), **res}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"[{i}] {row['agent']:32s} {task.spec_id or task.kind:12s} reward={res['reward']:.3f}"
                      f" ({row['secs']}s)")
        del agent
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    summary = summarize(results)
    if not args.results:
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_summary(args) -> None:
    """Merge parallel bench parts (<out>/parts/*.jsonl) into results.jsonl and write summary.json."""
    out = Path(args.out)
    parts = sorted((out / "parts").glob("*.jsonl"))
    if parts:
        (out / "results.jsonl").write_text("".join(part.read_text() for part in parts))
    summary = summarize(out / "results.jsonl")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v["mean"] for k, v in summary.items()}, indent=2))


def cmd_rescore(args) -> None:
    """Recompute rewards in a bench folder by replaying each row's stored edits (after a reward change)."""
    env = VitalEnv()
    out = Path(args.out)
    tasks = load_tasks(out / "tasks.jsonl")
    results = out / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    renders = args.renders or (1 if all(preset_name(t.state).startswith("procedural-") for t in tasks) else 3)
    for row in rows:
        if "turns" not in row:
            continue
        task = tasks[row["i"]]
        env.load_state(task.state)
        for turn in row["turns"]:
            env.apply(turn["edits"])
        row.update(score_state(env, task, env.state, renders))
    results.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = summarize(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v["mean"] for k, v in summary.items()}, indent=2))


def cmd_demo(args) -> None:
    """Save full trajectories (prompt, raw reply, parameter changes, audio per round) for showcase."""
    rng = random.Random(args.seed)
    env = VitalEnv()
    agent = make_agent(args.agent, False)
    root = Path(args.out)
    for n, spec in enumerate(args.specs, 1):
        args_one = argparse.Namespace(presets=args.presets, kind="edit", specs=[spec], n=1)
        task = make_tasks(env, args_one, rng)[0]
        out = root / f"{n:02d}-{spec}"
        out.mkdir(parents=True, exist_ok=True)
        before = describe(env.render(task.state))
        result = run_episode(env, agent, task, args.rounds, out)
        after = describe(env.render(env.state))
        record = {"agent": agent.name, "preset": preset_name(task.state), "instruction": task.instruction,
                  "spec": task.spec_id, "system": SYSTEM, "before": before, "after": after, **result}
        (out / "trajectory.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
        print(f"{out.name}: reward={result['reward']:.3f} preset={record['preset']}")


def cmd_noise(args) -> None:
    """Descriptor spread over repeated renders of the same state (reward noise floor)."""
    env = VitalEnv()
    for path in list_presets(args.presets)[: args.n]:
        state = env.load_preset(path)
        feats = [describe(env.render(state)) for _ in range(args.repeats)]
        spread = {k: round(statistics.pstdev(f[k] for f in feats), 4) for k in feats[0]}
        print(path.stem, spread)


def cmd_export(args) -> None:
    """Write single-turn RL prompts: starting state, instruction, and rendered audio paths."""
    env = VitalEnv()
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    tasks = make_tasks(env, args, random.Random(args.seed))
    with (out / "tasks.jsonl").open("w") as f:
        for i, task in enumerate(tasks):
            audio_path = out / "audio" / f"{i:06d}.wav"
            sf.write(audio_path, env.render(task.state).T, SR)
            row = {"id": i, "kind": task.kind, "spec_id": task.spec_id, "instruction": task.instruction,
                   "state": task.state, "target_state": task.target_state, "audio": str(audio_path)}
            if task.target_state:
                target_path = out / "audio" / f"{i:06d}_target.wav"
                sf.write(target_path, env.render(task.target_state).T, SR)
                row["target_audio"] = str(target_path)
            env.load_state(task.state)
            row["params"] = env.params()
            row["system"] = SYSTEM
            row["prompt"] = prompt_text(row["params"], task.instruction)
            if task.target_state:
                row["prompt"] += "\n\nSecond clip: target sound."
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(tasks)} tasks to {out / 'tasks.jsonl'}")


def main() -> None:
    p = argparse.ArgumentParser(prog="synth-rl")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("edit", help="edit a preset from a natural-language instruction")
    e.add_argument("--preset", required=True, help=".vital file")
    e.add_argument("--instruction", required=True, help='for example "brighter, with a shorter tail"')
    e.add_argument("--spec", choices=sorted(EDIT_SPECS), help="instruction family to score the result against "
                                                             "(optional; free-form instructions are not scored)")
    e.add_argument("--agent", default="gemini-3.8-flash",
                   help="a Gemini model id, a local Qwen Omni folder, 'folder::lora_adapter', or 'rule' (keywords)")
    e.add_argument("--rounds", type=int, default=2)
    e.add_argument("--mute-audio", action="store_true", help="send silence instead of the render")
    e.add_argument("--out", default="outputs/edit")
    e.set_defaults(func=cmd_edit)

    t = sub.add_parser("match", help="build a patch that sounds like a recording")
    t.add_argument("--target", required=True, help="audio file: one note at middle C (C4), up to 3 s")
    t.add_argument("--preset", help="start from this .vital instead of the init patch")
    t.add_argument("--agent", default="gemini-3.8-flash",
                   help="a Gemini model id, a local Qwen Omni folder, or 'folder::lora_adapter'")
    t.add_argument("--rounds", type=int, default=3)
    t.add_argument("--mute-audio", action="store_true", help="send silence instead of the render")
    t.add_argument("--out", default="outputs/match")
    t.set_defaults(func=cmd_match)

    b = sub.add_parser("bench", help="score agents on random tasks")
    b.add_argument("--presets", default=DEFAULT_PRESETS, help="'procedural' (generated patches, deterministic renders) or a folder of .vital presets")
    b.add_argument("--agents", nargs="+", default=["rule"],
                   help="'rule', local Qwen Omni folders ('folder::lora_adapter' for a fine-tune), Gemini model ids")
    b.add_argument("--kind", choices=["edit", "repair", "match"], default="edit",
                   help="edit: follow an instruction; repair: undo a few randomized parameters; match: rebuild a "
                        "target sound from the init patch")
    b.add_argument("--specs", nargs="*", default=None,
                   help="instruction families to sample; 'mono' drops the stereo-width ones")
    b.add_argument("--n", type=int, default=20)
    b.add_argument("--tasks", help="task file to reuse (default: <out>/tasks.jsonl, created if missing)")
    b.add_argument("--rounds", type=int, default=1)
    b.add_argument("--renders", type=int, default=0, help="renders averaged per score (default 1 procedural, 3 presets)")
    b.add_argument("--mute-audio", action="store_true", help="send silence instead of the render")
    b.add_argument("--blind", action="store_true", help="hide current parameter values; the agent must listen")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="outputs/bench")
    b.add_argument("--results", help="results file to append to (default <out>/results.jsonl); for parallel runs, "
                                     "write <out>/parts/<name>.jsonl and merge them with 'summary'")
    b.set_defaults(func=cmd_bench)

    m = sub.add_parser("summary", help="merge parallel bench parts and summarize a bench folder")
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_summary)

    r = sub.add_parser("rescore", help="recompute rewards of a bench folder from its stored edits")
    r.add_argument("--out", required=True)
    r.add_argument("--renders", type=int, default=0)
    r.set_defaults(func=cmd_rescore)

    d = sub.add_parser("demo", help="save showcase trajectories, one per requested instruction family")
    d.add_argument("--presets", default=DEFAULT_PRESETS, help="'procedural' (generated patches, deterministic renders) or a folder of .vital presets")
    d.add_argument("--agent", default="gemini-3.8-flash")
    d.add_argument("--specs", nargs="+", default=["brighter", "plucky", "spacious", "wider"])
    d.add_argument("--rounds", type=int, default=2)
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--out", default="outputs/demo")
    d.set_defaults(func=cmd_demo)

    n = sub.add_parser("noise", help="descriptor spread over repeated renders")
    n.add_argument("--presets", default="~/Music/Vital")
    n.add_argument("--n", type=int, default=5)
    n.add_argument("--repeats", type=int, default=5)
    n.set_defaults(func=cmd_noise)

    x = sub.add_parser("export", help="write single-turn RL prompts with rendered audio")
    x.add_argument("--presets", default=DEFAULT_PRESETS, help="'procedural' (generated patches, deterministic renders) or a folder of .vital presets")
    x.add_argument("--kind", choices=["edit", "repair", "match"], default="edit")
    x.add_argument("--specs", nargs="*", default=None, help="instruction families; 'mono' drops stereo width")
    x.add_argument("--n", type=int, default=100)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--out", default="data/rl")
    x.set_defaults(func=cmd_export)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
