"""Reward adapters for RL frameworks.

Each exported task row (see `synth-rl export`) carries the starting preset state and task metadata.
A completion is parsed as JSON edits, applied to that state, rendered, and scored. Completions
without any valid edit get -3: edit rewards are bounded below by about -2.4 (target -1, drift -0.5,
loudness -0.3, sparsity -0.64 for all 36 parameters) and repair and match rewards by -1, so format
failures rank below every real attempt.
"""

from __future__ import annotations

import json

from .agents import parse_edits
from .tasks import Task, score
from .vital_env import EDITABLE, VitalEnv

FORMAT_FAIL = -3.0
_env: VitalEnv | None = None


def _get_env() -> VitalEnv:
    # One Vital instance per worker process.
    global _env
    if _env is None:
        _env = VitalEnv()
    return _env


def reward_from_row(row: dict, completion: str) -> float:
    edits = {k: v for k, v in parse_edits(completion).items() if k in EDITABLE}
    if not edits:
        return FORMAT_FAIL
    task = Task(row["kind"], row["state"], row["instruction"], spec_id=row.get("spec_id", ""),
                target_state=row.get("target_state", ""))
    return float(score(_get_env(), task, edits)["reward"])


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """verl-style custom reward function; the task row is passed as extra_info or JSON ground_truth."""
    row = extra_info if extra_info else json.loads(ground_truth)
    return reward_from_row(row, solution_str)


class SynthEditORM:
    """ms-swift-style outcome reward: dataset columns arrive as keyword lists aligned with completions.

    Untested against ms-swift; register with `orms["synth_edit"] = SynthEditORM` in an external plugin.
    """

    def __call__(self, completions, kind, state, instruction, spec_id=None, target_state=None, **kwargs):
        n = len(completions)
        spec_id = spec_id or [""] * n
        target_state = target_state or [""] * n
        return [
            reward_from_row({"kind": kind[i], "state": state[i], "instruction": instruction[i],
                             "spec_id": spec_id[i], "target_state": target_state[i]}, completions[i])
            for i in range(n)
        ]
