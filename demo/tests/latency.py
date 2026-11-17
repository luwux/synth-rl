"""Latency of the OpenRouter listening agent under different request options (5 calls each by default).

    zsh -ic 'cd <repo> && .venv/bin/python demo/tests/latency.py --n 5 baseline off off+json'

Each call: procedural patch seed 3, "brighter, with a shorter tail". Prints seconds, completion and reasoning
tokens, and whether the edit was correct (centroid_oct up, tail_s not up).
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from synth_rl.agents import OpenAIChatAgent
from synth_rl.features import describe
from synth_rl.procedural import random_patch
from synth_rl.vital_env import VitalEnv

VARIANTS = {
    "baseline": {},
    "off": {"reasoning": {"enabled": False}},
    "off+cap": {"reasoning": {"enabled": False}, "max_tokens": 400},
    "off+json": {"reasoning": {"enabled": False}, "max_tokens": 400, "response_format": {"type": "json_object"}},
    "effort-low": {"reasoning": {"effort": "low"}},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+", choices=list(VARIANTS))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="qwen/qwen3.8-omni-flash")
    args = ap.parse_args()
    env = VitalEnv()
    state = random_patch(env, random.Random(3))
    env.load_state(state)
    audio = env.render(state)
    params = env.params()
    before = describe(audio)
    for variant in args.variants:
        agent = OpenAIChatAgent(args.model, options=VARIANTS[variant])
        secs, ok = [], 0
        for i in range(args.n):
            t = time.time()
            try:
                out = agent.propose("brighter, with a shorter tail", params, audio)
            except Exception as e:  # noqa: BLE001
                print(f"{variant} call {i + 1}: error {e}")
                continue
            dt = time.time() - t
            secs.append(dt)
            env.load_state(state)
            env.apply(out["edits"])
            after = describe(env.render(env.state))
            good = bool(out["edits"]) and after["centroid_oct"] > before["centroid_oct"] and after["tail_s"] <= before["tail_s"]
            ok += good
            u = agent.last_usage
            reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
            print(f"{variant} call {i + 1}: {dt:5.1f} s  completion {u.get('completion_tokens')}  reasoning {reasoning}"
                  f"  correct {good}  edits {sorted(out['edits'])}")
        if secs:
            print(f"== {variant}: median {statistics.median(secs):.1f} s, mean {statistics.mean(secs):.1f} s, "
                  f"min {min(secs):.1f}, max {max(secs):.1f}, correct {ok}/{args.n}")


if __name__ == "__main__":
    main()
