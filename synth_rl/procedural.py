"""Procedurally generated Vital patches, so tasks need no third-party preset library.

Patches start from the init patch (Vital's init preset with an envelope-2-to-cutoff routing at zero amount) and
randomize only editable parameters: oscillators, filter, envelopes, the filter envelope amount, and effects. So any
generated patch can be rebuilt from the init patch through the agent's action space. Silent or clipping patches
are resampled.
"""

from __future__ import annotations

import json
import random

import numpy as np

from .vital_env import FILTER_ENV, VitalEnv, fixed_phase


def _unison(voices: int) -> float:
    return (voices - 1) / 15


def init_patch(env: VitalEnv) -> str:
    """Starting point for building a sound from scratch."""
    env.synth.load_init_preset()
    env.synth.connect_modulation(*FILTER_ENV)
    data = json.loads(fixed_phase(env.synth.to_json()))
    data["preset_name"] = "init"
    env.load_state(json.dumps(data))
    return env.state


def random_patch(env: VitalEnv, rng: random.Random, attempts: int = 10) -> str:
    for _ in range(attempts):
        synth = env.synth
        synth.load_init_preset()
        synth.connect_modulation(*FILTER_ENV)
        c = synth.get_controls()
        u = rng.uniform

        def put(name: str, value: float) -> None:
            c[name].set_normalized(float(np.clip(value, 0.0, 1.0)))

        put("osc_1_level", u(0.5, 0.9))
        put("osc_1_wave_frame", u(0, 1))
        put("osc_1_unison_voices", _unison(rng.choice([1, 1, 2, 3, 5, 7])))
        put("osc_1_unison_detune", u(0, 0.5))
        put("osc_1_stereo_spread", u(0, 1))
        if rng.random() < 0.5:
            put("osc_2_on", 1)
            put("osc_2_level", u(0.3, 0.8))
            put("osc_2_wave_frame", u(0, 1))
            put("osc_2_unison_detune", u(0, 0.5))
        if rng.random() < 0.75:
            put("filter_1_on", 1)
            put("filter_1_cutoff", u(0.25, 0.9))
            put("filter_1_resonance", u(0, 0.6))
            put("filter_1_drive", u(0, 0.3))
        put("env_1_attack", u(0, 0.45) ** 1.5)
        put("env_1_decay", u(0.2, 0.8))
        put("env_1_sustain", rng.choice([u(0, 0.3), u(0.5, 1)]))
        put("env_1_release", u(0.1, 0.6))
        put("env_2_attack", u(0, 0.3))
        put("env_2_decay", u(0.2, 0.7))
        put("env_2_sustain", u(0, 1))
        if rng.random() < 0.5:
            c["modulation_1_amount"].set_normalized(u(0.55, 0.8))
        for fx, p, mix in (("chorus", 0.3, (0.2, 0.6)), ("reverb", 0.5, (0.1, 0.5)),
                           ("delay", 0.2, (0.1, 0.4)), ("distortion", 0.2, (0.3, 0.8))):
            if rng.random() < p:
                put(f"{fx}_on", 1)
                put(f"{fx}_dry_wet" if fx != "distortion" else "distortion_mix", u(*mix))
        if c["distortion_on"].value() > 0.5:
            put("distortion_drive", u(0.5, 0.75))
        if c["reverb_on"].value() > 0.5:
            put("reverb_decay_time", u(0.2, 0.7))
            put("reverb_size", u(0.3, 0.9))

        state = fixed_phase(synth.to_json())
        data = json.loads(state)
        data["preset_name"] = f"procedural-{rng.getrandbits(32):08x}"
        state = json.dumps(data)
        audio = env.render(state)
        peak = float(np.abs(audio).max())
        if 1e-3 < peak < 1.5:
            env.load_state(state)
            return state
    raise RuntimeError("could not generate an audible patch")
