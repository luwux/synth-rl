"""Headless Vital environment: load presets, edit a curated parameter subset, render audio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import vita

SR = 44100
NOTE = 60
VELOCITY = 0.8
NOTE_DUR = 1.5
RENDER_DUR = 3.0

# Parameters an agent may edit, with the description shown in prompts. Vital's env_1 is the amp envelope.
EDITABLE = {
    "osc_1_level": "oscillator 1 level",
    "osc_1_wave_frame": "oscillator 1 wavetable position",
    "osc_1_unison_voices": "oscillator 1 unison voice count",
    "osc_1_unison_detune": "oscillator 1 unison detune amount",
    "osc_1_stereo_spread": "oscillator 1 unison stereo spread",
    "osc_2_on": "oscillator 2 switch",
    "osc_2_level": "oscillator 2 level",
    "osc_2_wave_frame": "oscillator 2 wavetable position",
    "osc_2_unison_detune": "oscillator 2 unison detune amount",
    "filter_1_on": "filter 1 switch",
    "filter_1_cutoff": "filter 1 cutoff",
    "filter_1_resonance": "filter 1 resonance",
    "filter_1_drive": "filter 1 drive",
    "filter_1_mix": "filter 1 dry/wet",
    "env_1_attack": "amp envelope attack",
    "env_1_decay": "amp envelope decay",
    "env_1_sustain": "amp envelope sustain",
    "env_1_release": "amp envelope release",
    "env_2_attack": "envelope 2 attack (shapes the filter through filter_env_amount)",
    "env_2_decay": "envelope 2 decay",
    "env_2_sustain": "envelope 2 sustain",
    "filter_env_amount": "envelope 2 to filter 1 cutoff amount (0.5 none; above opens the filter on each note)",
    "chorus_on": "chorus switch",
    "chorus_dry_wet": "chorus mix",
    "reverb_on": "reverb switch",
    "reverb_dry_wet": "reverb mix",
    "reverb_decay_time": "reverb decay time (0 short, 1 long)",
    "reverb_size": "reverb size",
    "delay_on": "delay switch",
    "delay_dry_wet": "delay mix",
    "distortion_on": "distortion switch",
    "distortion_drive": "distortion drive",
    "distortion_mix": "distortion mix",
    "eq_on": "EQ switch",
    "eq_low_gain": "EQ low shelf gain",
    "eq_high_gain": "EQ high shelf gain",
}


# Vital's display text for these is wrong through vita (reverb decay reads 64 to 262144 secs), so prompts omit it.
NO_DISPLAY = {"reverb_decay_time"}
# `filter_env_amount` is the amount of whichever modulation slot routes env_2 to filter_1_cutoff. Every loaded
# state gets that routing (at zero amount, which does not change the sound), so the control always exists.
FILTER_ENV = ("env_2", "filter_1_cutoff")


def fixed_phase(state: str) -> str:
    """Turn off oscillator random start phase, Vital's default. With it off (and a fresh instance per render),
    patches without random modulators render bit-identically; presets that use them still vary."""
    data = json.loads(state)
    settings = data["settings"]
    for key in settings:
        if key.endswith("random_phase"):
            settings[key] = 0.0
    return json.dumps(data)


class VitalEnv:
    """One Vital instance. State is the full preset JSON string; edits use normalized 0–1 values."""

    def __init__(self) -> None:
        self.synth = vita.Synth()
        self.synth.set_sample_rate(SR)
        self.state = ""
        self.filter_env_slot: int | None = None

    def load_preset(self, path: str | Path) -> str:
        if not self.synth.load_preset(str(path)):
            raise ValueError(f"Vital could not load preset: {path}")
        self.load_state(fixed_phase(self.synth.to_json()))
        return self.state

    def load_state(self, state: str) -> None:
        self.synth.load_json(state)
        self.state = state
        self.filter_env_slot = self._filter_env_slot(state)
        if self.filter_env_slot is None and self.synth.connect_modulation(*FILTER_ENV):
            self.state = self.synth.to_json()
            self.filter_env_slot = self._filter_env_slot(self.state)

    @staticmethod
    def _filter_env_slot(state: str) -> int | None:
        mods = json.loads(state)["settings"].get("modulations", [])
        for i, m in enumerate(mods):
            if (m.get("source"), m.get("destination")) == FILTER_ENV:
                return i + 1
        return None

    def control_name(self, name: str) -> str | None:
        """Vital control behind an editable name; None when this state has no free slot for the filter envelope."""
        if name == "filter_env_amount":
            return None if self.filter_env_slot is None else f"modulation_{self.filter_env_slot}_amount"
        return name

    def values(self) -> dict[str, float]:
        """Exact normalized values of the editable parameters in the current state."""
        self.synth.load_json(self.state)
        controls = self.synth.get_controls()
        return {name: controls[self.control_name(name)].get_normalized() for name in EDITABLE if self.control_name(name)}

    def params(self) -> dict[str, dict]:
        """Current editable parameters: normalized value and Vital's display text."""
        self.synth.load_json(self.state)
        controls = self.synth.get_controls()
        out = {}
        for name, desc in EDITABLE.items():
            control = self.control_name(name)
            if control is None:
                continue
            out[name] = {
                "value": round(controls[control].get_normalized(), 3),
                "display": "" if name in NO_DISPLAY else self.synth.get_control_text(control).strip(),
                "desc": desc,
            }
        return out

    def apply(self, edits: dict[str, float]) -> list[str]:
        """Set normalized values for editable parameters; returns the names actually applied."""
        self.synth.load_json(self.state)
        controls = self.synth.get_controls()
        applied = []
        for name, value in edits.items():
            control = self.control_name(name) if name in EDITABLE else None
            if control is None:
                continue
            controls[control].set_normalized(float(np.clip(float(value), 0.0, 1.0)))
            applied.append(name)
        self.state = self.synth.to_json()
        return applied

    def render(self, state: str | None = None) -> np.ndarray:
        """Render one held note as stereo float32 (2, samples).

        Each render uses a fresh Vital instance: a reused one carries effect buffers, LFO phases, and
        random-generator state from the previous render, so the same state would sound different each time.
        """
        synth = vita.Synth()
        synth.set_sample_rate(SR)
        synth.load_json(state if state is not None else self.state)
        return np.asarray(synth.render(NOTE, VELOCITY, NOTE_DUR, RENDER_DUR), dtype=np.float32)


def load_audio(path: str | Path) -> np.ndarray:
    """Read a recording as stereo float32 (2, samples) at SR, cut or padded to the render length."""
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, rate = sf.read(str(Path(path).expanduser()), dtype="float32", always_2d=True)
    audio = audio.T[:2]
    if len(audio) == 1:
        audio = np.repeat(audio, 2, axis=0)
    if rate != SR:
        audio = resample_poly(audio, SR, rate, axis=1).astype(np.float32)
    n = int(RENDER_DUR * SR)
    return np.pad(audio[:, :n], ((0, 0), (0, max(0, n - audio.shape[1]))))


def list_presets(root: str | Path) -> list[Path]:
    return sorted(Path(root).expanduser().rglob("*.vital"))


def preset_name(state: str) -> str:
    return json.loads(state).get("preset_name", "")
