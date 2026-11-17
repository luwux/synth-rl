"""Two listening probes with ground truth from the synthesizer (the "direct listening check" in the README).

1. One knob wrong: the current sound differs from the target in exactly one parameter. One reply, real audio against
   silent clips. Scored in parameters: was the wrong knob among the ones set, how far was it moved back, how many
   right knobs were moved.
2. A model as judge: two clips that differ in one parameter by a known amount; "which clip is brighter?" and the like.
   Real audio against silent clips (chance is 50 %).

    python scripts/listening_probe.py [n_tasks] [pairs_per_attribute]      # defaults 60 and 24

Needs OPENROUTER_API_KEY. Writes every reply to outputs/listening/.
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth_rl.agents import OpenAIChatAgent, first_json
from synth_rl.procedural import random_patch
from synth_rl.tasks import make_repair_task
from synth_rl.vital_env import VitalEnv, from_unit, moved

MODEL = "qwen/qwen3.8-omni-flash"
OUT = Path("outputs/listening")
OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
PAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
env = VitalEnv()
agents = {"audio": OpenAIChatAgent(MODEL), "muted": OpenAIChatAgent(MODEL, mute_audio=True)}
boot = np.random.default_rng(0)


def ci(values) -> str:
    v = np.asarray(values, dtype=float)
    b = [v[boot.integers(0, len(v), len(v))].mean() for _ in range(4000)]
    return f"{v.mean():.2f} [{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}]"


# --- 1. one knob wrong ---------------------------------------------------------------------------------------
rng = random.Random(20261116)
tasks = []
while len(tasks) < N:
    task = make_repair_task(env, random_patch(env, rng), rng, k=(1, 1))
    if task:
        tasks.append(task)
jobs = []
for i, task in enumerate(tasks):
    env.load_state(task.state)
    params, audio, target = env.params(), env.render(task.state), env.render(task.target_state)
    jobs += [(i, cond, task.instruction, params, audio, target) for cond in agents]


def propose(job):
    i, cond, instruction, params, audio, target = job
    for _ in range(4):
        try:
            return i, cond, agents[cond].propose(instruction, params, audio, target)
        except RuntimeError:
            pass
    return i, cond, None


with ThreadPoolExecutor(8) as pool:
    replies = list(pool.map(propose, jobs))
rows = []
for i, cond, out in replies:
    if out is None:
        continue
    task = tasks[i]
    env.load_state(task.target_state)
    target = env.values()
    env.load_state(task.state)
    start = env.values()
    wrong = moved(start, target, 1e-3)
    env.apply(out["edits"])
    now = env.values()
    touched = moved(start, now, 1e-3)
    restored = float(np.mean([max(-1.0, 1 - abs(now[n] - target[n]) / abs(start[n] - target[n])) for n in wrong]))
    rows.append({"task": i, "cond": cond, "knob": task.perturbed[0], "wrong": wrong, "touched": touched,
                 "hit": any(n in wrong for n in touched), "restored": restored, "others": sum(n not in wrong for n in touched),
                 "raw": out["raw"]})
(OUT / "one_knob.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"=== 1. one knob wrong: {N} tasks, one reply each ===")
by = {c: {r["task"]: r for r in rows if r["cond"] == c} for c in agents}
ids = sorted(set(by["audio"]) & set(by["muted"]))
for c in agents:
    R = [by[c][i] for i in ids]
    print(f"{c:6s} set the wrong knob {ci([r['hit'] for r in R])} | moved it back by half or more {ci([r['restored'] >= 0.5 for r in R])} | "
          f"mean restored {ci([r['restored'] for r in R])} | right knobs moved {ci([r['others'] for r in R])} | "
          f"only the wrong knob, at least half-way {ci([r['restored'] >= 0.5 and r['others'] == 0 for r in R])}")
print("paired audio - muted: set the wrong knob", ci([by["audio"][i]["hit"] - by["muted"][i]["hit"] for i in ids]),
      "| moved back by half or more", ci([(by["audio"][i]["restored"] >= 0.5) - (by["muted"][i]["restored"] >= 0.5) for i in ids]))
group = lambda n: n.split("_")[0] if not n.startswith(("osc", "env", "filter")) else "_".join(n.split("_")[:2])
print("by section of the wrong knob (tasks: found with audio / found muted):",
      {g: f"{sum(1 for i in ids if group(by['audio'][i]['knob']) == g)}: {sum(by['audio'][i]['hit'] for i in ids if group(by['audio'][i]['knob']) == g)} / {sum(by['muted'][i]['hit'] for i in ids if group(by['muted'][i]['knob']) == g)}"
       for g in sorted({group(by['audio'][i]['knob']) for i in ids})})

# --- 2. a model as judge -------------------------------------------------------------------------------------
# attribute: (question, parameter, low and high values in the parameter's unit for a large and a small difference, fixed settings)
ATTRIBUTES = {
    "brighter": ("Which clip is brighter (more high-frequency content)?", "filter_1_cutoff", {"large": (500, 4000), "small": (1200, 2000)}, {"filter_1_on": 1.0}),
    "more reverb": ("Which clip has more reverb?", "reverb_dry_wet", {"large": (5, 60), "small": (20, 35)}, {"reverb_on": 1.0}),
    "shorter": ("Which clip dies away sooner while the key is still held (more like a pluck)?", "env_1_decay", {"large": (0.15, 2.5), "small": (0.4, 0.8)}, {"env_1_sustain": 0.0}),
    "more detuned": ("Which clip is more detuned (thicker, more beating between voices)?", "osc_1_unison_detune", {"large": (2, 45), "small": (10, 20)}, {"osc_1_unison_voices": 4 / 15}),
    "more distorted": ("Which clip is more distorted?", "distortion_drive", {"large": (-5, 20), "small": (5, 10)}, {"distortion_on": 1.0, "distortion_mix": 1.0}),
    "louder": ("Which clip is louder?", "osc_1_level", {"large": (20, 90), "small": (50, 70)}, {}),
}
SYSTEM = ('You compare two audio clips of one synthesizer note. Listen to both, then answer with JSON only: '
          '{"answer": "first"} or {"answer": "second"}.')
rng = random.Random(7)
pairs = []
for attribute, (question, name, levels, fixed) in ATTRIBUTES.items():
    for level, (low, high) in levels.items():
        for _ in range(PAIRS // 2):
            env.load_state(random_patch(env, rng))
            clips = []
            for value in (low, high):
                env.apply({**fixed, name: from_unit(name, value)})
                clips.append(env.render(env.state))
            first_is_more = rng.random() < 0.5
            a, b = (clips[1], clips[0]) if first_is_more else (clips[0], clips[1])
            # for "shorter" the lower value is the one asked for
            truth = ("second" if first_is_more else "first") if attribute == "shorter" else ("first" if first_is_more else "second")
            pairs.append((attribute, level, question, a, b, truth))


def judge(job):
    k, cond = job
    attribute, level, question, a, b, truth = pairs[k]
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": "First clip:"}, {"type": "audio", "audio": a},
                                             {"type": "text", "text": "Second clip:"}, {"type": "audio", "audio": b},
                                             {"type": "text", "text": question}]}]
    for _ in range(4):
        try:
            answer = str((first_json(agents[cond].complete(messages)) or {}).get("answer", "")).lower()
            return {"attribute": attribute, "level": level, "cond": cond, "truth": truth, "answer": answer, "correct": answer == truth}
        except RuntimeError:
            pass
    return None


with ThreadPoolExecutor(8) as pool:
    answers = [r for r in pool.map(judge, [(k, c) for k in range(len(pairs)) for c in agents]) if r]
(OUT / "judge.jsonl").write_text("".join(json.dumps(r) + "\n" for r in answers))
print(f"\n=== 2. the model as judge: {PAIRS // 2} pairs per attribute and difference size; share correct [95 % interval]; chance 0.50 ===")
for attribute in ATTRIBUTES:
    line = f"{attribute:15s}"
    for level in ("large", "small"):
        for cond in agents:
            R = [r["correct"] for r in answers if (r["attribute"], r["level"], r["cond"]) == (attribute, level, cond)]
            line += f" | {level} {cond} {ci(R)}"
    print(line)
for cond in agents:
    R = [r for r in answers if r["cond"] == cond]
    print(f"all attributes, {cond}: {ci([r['correct'] for r in R])} | answered 'first' in {np.mean([r['answer'] == 'first' for r in R]):.2f} of replies, unreadable {np.mean([r['answer'] not in ('first', 'second') for r in R]):.2f}")
