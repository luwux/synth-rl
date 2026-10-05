"""Task generators and rewards.

Edit task: natural-language instruction; reward from the signed change of target descriptors,
minus drift in descriptors the instruction did not ask to change.
Repair task: a preset with a few parameters replaced by values from another patch; reward from how much
closer the render gets to the original preset's render.
Match task: rebuild a target sound from the init patch (sound matching from scratch); same reward as repair.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .features import describe, mss_distance
from .vital_env import EDITABLE, VitalEnv

# Change of one unit on this scale counts as a clearly audible step.
SCALE = {
    "centroid_oct": 0.5,
    "hf_db": 6.0,
    "loudness_db": 6.0,
    "attack_s": 0.1,
    "sustain": 0.25,
    "tail_s": 0.4,
    "width": 0.1,
}

# id -> paraphrases, targeted descriptors with sign, descriptors that must stay put,
# and a feasibility check on the starting descriptors.
EDIT_SPECS = {
    "brighter": {
        "phrases": ["make it brighter", "open up the top end", "more sparkle and air", "less muffled"],
        "target": {"centroid_oct": 1, "hf_db": 1},
        "keep": ["attack_s", "sustain", "tail_s"],
        "feasible": lambda f: f["centroid_oct"] < 12.5,
    },
    "darker": {
        "phrases": ["make it darker", "tame the highs", "warmer and more muffled", "less harsh"],
        "target": {"centroid_oct": -1, "hf_db": -1},
        "keep": ["attack_s", "sustain", "tail_s"],
        "feasible": lambda f: f["centroid_oct"] > 8.5,
    },
    "plucky": {
        "phrases": ["make it more plucky", "shorter, pluckier notes", "turn it into a pluck"],
        "target": {"sustain": -1},
        "keep": ["centroid_oct", "width"],
        "feasible": lambda f: f["sustain"] > 0.3,
    },
    "sustained": {
        "phrases": ["make it sustain while the key is held", "more pad-like, less decay", "hold the note longer"],
        "target": {"sustain": 1},
        "keep": ["centroid_oct", "width"],
        "feasible": lambda f: f["sustain"] < 0.6,
    },
    "soft_attack": {
        "phrases": ["give it a slower, softer attack", "fade the notes in", "swell in instead of hitting"],
        "target": {"attack_s": 1},
        "keep": ["centroid_oct", "width"],
        "feasible": lambda f: f["attack_s"] < 0.3,
    },
    "wider": {
        "phrases": ["make it wider in stereo", "spread it out across the stereo field", "more stereo width"],
        "target": {"width": 1},
        "keep": ["centroid_oct", "attack_s", "sustain"],
        "feasible": lambda f: f["width"] < 0.3,
    },
    "narrower": {
        "phrases": ["make it narrower, closer to mono", "collapse the stereo width"],
        "target": {"width": -1},
        "keep": ["centroid_oct", "attack_s", "sustain"],
        "feasible": lambda f: f["width"] > 0.05,
    },
    "spacious": {
        "phrases": ["add more space and reverb", "make it sound farther away in a big room", "longer, washier tail"],
        "target": {"tail_s": 1},
        "keep": ["centroid_oct", "attack_s"],
        "feasible": lambda f: f["tail_s"] < 1.2,
    },
    "drier": {
        "phrases": ["make it drier, less reverb", "bring it up close and dry", "shorten the tail"],
        "target": {"tail_s": -1},
        "keep": ["centroid_oct", "attack_s"],
        "feasible": lambda f: f["tail_s"] > 0.3,
    },
}
# 16 kHz mono models cannot hear stereo width.
MONO_SPECS = [k for k in EDIT_SPECS if k not in ("wider", "narrower")]


REPAIR_INSTRUCTION = "The second clip is the target sound. Adjust the parameters so the current sound matches it."
MATCH_INSTRUCTION = ("The second clip is the target sound. The current sound is a basic init patch: "
                     "set the parameters so it matches the target.")


@dataclass
class Task:
    kind: str  # "edit", "repair", or "match"
    state: str  # starting preset JSON
    instruction: str
    spec_id: str = ""
    target_state: str = ""  # repair and match
    perturbed: list[str] = field(default_factory=list)


def make_edit_task(env: VitalEnv, state: str, rng: random.Random, specs: list[str] | None = None) -> Task | None:
    feats = describe(env.render(state))
    options = [k for k, s in EDIT_SPECS.items() if s["feasible"](feats) and (not specs or k in specs)]
    if not options:
        return None
    spec_id = rng.choice(options)
    return Task("edit", state, rng.choice(EDIT_SPECS[spec_id]["phrases"]), spec_id=spec_id)


def make_repair_task(env: VitalEnv, state: str, rng: random.Random, k: tuple[int, int] = (2, 4),
                     min_dist: float = 0.1, attempts: int = 20) -> Task | None:
    """Copy 2-4 values from another procedural patch, so the broken values look no less plausible than the
    rest (no text-only shortcut), and retry until the change is audible (an inaudible change makes every
    edit score -1). min_dist is about the distance of a small level change."""
    from .procedural import random_patch

    target = env.render(state)
    env.load_state(state)
    current = env.values()
    for _ in range(attempts):
        env.load_state(random_patch(env, rng))
        donor = env.values()
        differ = [n for n in sorted(current) if abs(donor[n] - current[n]) > 1e-3]
        names = rng.sample(differ, min(len(differ), rng.randint(*k)))
        env.load_state(state)
        env.apply({n: donor[n] for n in names})
        if mss_distance(env.render(env.state), target) >= min_dist:
            return Task("repair", env.state, REPAIR_INSTRUCTION, target_state=state, perturbed=names)
    return None


def make_match_task(env: VitalEnv, target_state: str) -> Task:
    from .procedural import init_patch

    return Task("match", init_patch(env), MATCH_INSTRUCTION, target_state=target_state)


def edit_reward(before: dict[str, float], after: dict[str, float], spec_id: str, n_changed: int) -> dict[str, float]:
    spec = EDIT_SPECS[spec_id]
    gains = [np.tanh(2 * sign * (after[f] - before[f]) / SCALE[f]) for f, sign in spec["target"].items()]
    drift = [min(1.0, abs(after[f] - before[f]) / SCALE[f]) for f in spec["keep"]]
    loud = min(1.0, max(0.0, abs(after["loudness_db"] - before["loudness_db"]) - 3.0) / SCALE["loudness_db"])
    target = float(np.mean(gains))
    keep = float(np.mean(drift)) if drift else 0.0
    sparsity = 0.02 * max(0, n_changed - 4)
    return {
        "reward": target - 0.5 * keep - 0.3 * loud - sparsity,
        "target": target,
        "drift": keep,
        "loudness_drift": loud,
    }


def repair_reward(before_dist: float, after_dist: float) -> dict[str, float]:
    improvement = (before_dist - after_dist) / (before_dist + 1e-9)
    return {"reward": float(np.tanh(3 * improvement)), "dist_before": before_dist, "dist_after": after_dist}


def changed_params(env: VitalEnv, a: str, b: str, tol: float = 1e-3) -> list[str]:
    env.load_state(a)
    pa = env.params()
    env.load_state(b)
    pb = env.params()
    return [k for k in EDITABLE if abs(pa[k]["value"] - pb[k]["value"]) > tol]


def mean_describe(env: VitalEnv, state: str, renders: int) -> dict[str, float]:
    feats = [describe(env.render(state)) for _ in range(renders)]
    return {k: float(np.mean([f[k] for f in feats])) for k in feats[0]}


def score_state(env: VitalEnv, task: Task, final_state: str, renders: int = 1) -> dict[str, float]:
    """Render the task's starting state and a final state, and score the change.

    `renders` > 1 averages descriptors over repeated renders, for presets whose LFOs or random
    modulators make renders differ. An unchanged state scores exactly as "no change".
    """
    n_changed = len(changed_params(env, task.state, final_state))
    if task.kind == "edit":
        before = mean_describe(env, task.state, renders)
        after = before if n_changed == 0 else mean_describe(env, final_state, renders)
        out = edit_reward(before, after, task.spec_id, n_changed)
    else:
        target_audio = env.render(task.target_state)
        before_dist = float(np.mean([mss_distance(env.render(task.state), target_audio) for _ in range(renders)]))
        after_dist = before_dist if n_changed == 0 else float(
            np.mean([mss_distance(env.render(final_state), target_audio) for _ in range(renders)]))
        out = repair_reward(before_dist, after_dist)
    out["n_changed"] = n_changed
    return out


def score_against_audio(env: VitalEnv, start_state: str, final_state: str, target: np.ndarray) -> dict[str, float]:
    """Match reward against a recorded target sound instead of a target preset."""
    n_changed = len(changed_params(env, start_state, final_state))
    before = mss_distance(env.render(start_state), target)
    after = before if n_changed == 0 else mss_distance(env.render(final_state), target)
    return {**repair_reward(before, after), "n_changed": n_changed}


def score(env: VitalEnv, task: Task, edits: dict[str, float], renders: int = 1) -> dict[str, float]:
    """Apply edits to the task's starting state, render, and score."""
    env.load_state(task.state)
    env.apply(edits)
    return score_state(env, task, env.state, renders)
