"""Headless Vital environment: load presets, edit a curated parameter subset, render audio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import vita

from .units import number_text, unit_of
from .wavetable import OSCILLATORS, POSITIONS, has_shapes, position_text

SR = 44100
NOTE = 60
VELOCITY = 0.8
NOTE_DUR = 1.5
RENDER_DUR = 3.0

CUTOFF = "filter_1_cutoff"
PEAK = "filter_1_cutoff_at_env_2_peak"

# Parameters an agent may edit, with the description shown in prompts. Each is one Vital control except PEAK,
# which stands for the depth of the envelope-2-to-cutoff routing. Vital's env_1 is the amp envelope.
EDITABLE = {
    "osc_1_level": "oscillator 1 level",
    "osc_1_wave_frame": "oscillator 1 wavetable position",
    "osc_1_unison_voices": "oscillator 1 unison voice count",
    "osc_1_unison_detune": "oscillator 1 detune between its unison voices (no effect with 1 voice)",
    "osc_1_stereo_spread": "oscillator 1 unison stereo spread",
    "osc_2_on": "oscillator 2 switch",
    "osc_2_level": "oscillator 2 level",
    "osc_2_wave_frame": "oscillator 2 wavetable position",
    "osc_2_unison_voices": "oscillator 2 unison voice count",
    "osc_2_unison_detune": "oscillator 2 detune between its unison voices (no effect with 1 voice)",
    "filter_1_on": "filter 1 switch",
    CUTOFF: "filter 1 cutoff while envelope 2 is at zero (its resting value)",
    PEAK: "filter 1 cutoff when envelope 2 is at its peak: envelope 2 moves the cutoff between filter_1_cutoff and "
          "this value (equal: no movement; higher: each note opens the filter; lower: each note closes it)",
    "filter_1_resonance": "filter 1 resonance",
    "filter_1_drive": "filter 1 drive",
    "filter_1_mix": "filter 1 dry/wet",
    "env_1_attack": "amp envelope attack time",
    "env_1_decay": "amp envelope decay time (from full level down to the sustain level)",
    "env_1_sustain": "amp envelope sustain level, held while the key is down",
    "env_1_release": "amp envelope release time (the fade after the key is released)",
    "env_2_attack": "envelope 2 attack time (envelope 2 moves the filter cutoff)",
    "env_2_decay": "envelope 2 decay time (from its peak down to its sustain level)",
    "env_2_sustain": "envelope 2 sustain level (0 %: the cutoff settles at filter_1_cutoff; 100 %: it stays at the peak value)",
    "chorus_on": "chorus switch",
    "chorus_dry_wet": "chorus mix",
    "reverb_on": "reverb switch",
    "reverb_dry_wet": "reverb mix",
    "reverb_decay_time": "reverb decay time",
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

# vita maps the 0-1 range of this control to 6..18, all at or beyond its real range of -6..6 (the exponent of the
# decay time in seconds), so every setting gave the longest decay, 64 s. Here 0-1 covers the real range instead.
RAW_RANGE = {"reverb_decay_time": (-6.0, 6.0)}
# Every loaded state gets this routing (at zero depth, which does not change the sound), so PEAK always exists.
FILTER_ENV = ("env_2", CUTOFF)
# The routing's depth is a fraction of the cutoff's whole range, so cutoff position + depth is the position the
# envelope's peak reaches. Below this depth (0.01 semitone) the cutoff counts as not moving.
STILL = 1e-4
# Results saved before 2026-11-16 name the depth itself: 0.5 none, 1 the whole range upward.
LEGACY_DEPTH = "filter_env_amount"


def _raw_scales() -> dict[str, tuple[float, float, str]]:
    synth = vita.Synth()
    out = {}
    for name in EDITABLE:
        d = synth.get_control_details(CUTOFF if name == PEAK else name)
        out[name] = (*RAW_RANGE.get(name, (d.min, d.max)), str(d.scale).rsplit(".", 1)[-1])
    return out


RAW_SCALE = _raw_scales()  # name -> (lowest raw value, highest, vita's scale name)


def to_raw(name: str, position: float) -> float:
    """Vital's raw value at a 0-1 position, as vita's set_normalized computes it (except RAW_RANGE)."""
    low, high, scale = RAW_SCALE[name]
    position = min(1.0, max(0.0, float(position)))
    if scale == "Quadratic":
        position = position**0.5
    raw = low + (high - low) * position
    return float(int(raw + 0.5)) if scale == "Indexed" else raw


def to_unit(name: str, position: float) -> float:
    """The value of a 0-1 position in the parameter's unit."""
    return unit_of(name).from_raw(to_raw(name, position))


def from_unit(name: str, value: float) -> float:
    """The 0-1 position of a value in the parameter's unit; values outside the range land on its ends."""
    low, high, scale = RAW_SCALE[name]
    position = (min(high, max(low, unit_of(name).to_raw(value))) - low) / (high - low)
    return position**2 if scale == "Quadratic" else position


def unit_range(name: str) -> tuple[float, float]:
    low, high, _ = RAW_SCALE[name]
    return unit_of(name).from_raw(low), unit_of(name).from_raw(high)


def range_text(name: str) -> str:
    unit, (low, high) = unit_of(name), unit_range(name)
    if not unit.unit:
        return "on or off"
    return f"{number_text(low)} to {number_text(high)} {unit.unit}"


def moved(before: dict[str, float], after: dict[str, float], tol: float = 1e-3) -> list[str]:
    """Parameters whose 0-1 position differs. A peak that only followed the cutoff, with no movement before or
    after, is not a change of its own."""
    names = [n for n in EDITABLE if n in before and n in after and abs(before[n] - after[n]) > tol]
    if PEAK in names and all(abs(v[PEAK] - v[CUTOFF]) <= tol for v in (before, after)):
        names.remove(PEAK)
    return names


VITAL_VERSION = "1.5.5"


def preset_file(state: str) -> str:
    """Preset text for a `.vital` file. vita stamps states with synth_version 99999.9.9, and Vital refuses those
    ("Preset was created with a newer version"), so files carry a released version instead."""
    data = json.loads(state)
    data["synth_version"] = VITAL_VERSION
    return json.dumps(data)


def read_normalized(control, name: str) -> float:
    if name in RAW_RANGE:
        low, high = RAW_RANGE[name]
        return float(np.clip((control.value() - low) / (high - low), 0.0, 1.0))
    return control.get_normalized()


def write_normalized(control, name: str, value: float) -> None:
    value = float(np.clip(float(value), 0.0, 1.0))
    if name in RAW_RANGE:
        low, high = RAW_RANGE[name]
        control.set(low + value * (high - low))
    else:
        control.set_normalized(value)


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
    """One Vital instance. State is the full preset JSON string; edits are 0-1 positions (see `to_unit`)."""

    def __init__(self) -> None:
        self.synth = vita.Synth()
        self.synth.set_sample_rate(SR)
        self.state = ""
        self.env_slot: int | None = None  # modulation slot of FILTER_ENV; None when a preset has no free slot

    def load_preset(self, path: str | Path) -> str:
        if not self.synth.load_preset(str(path)):
            raise ValueError(f"Vital could not load preset: {path}")
        self.load_state(fixed_phase(self.synth.to_json()))
        return self.state

    def load_state(self, state: str) -> None:
        self.synth.load_json(state)
        self.state = state
        self.env_slot = self._env_slot(state)
        if self.env_slot is None and self.synth.connect_modulation(*FILTER_ENV):
            self.state = self.synth.to_json()
            self.env_slot = self._env_slot(self.state)

    @staticmethod
    def _env_slot(state: str) -> int | None:
        mods = json.loads(state)["settings"].get("modulations", [])
        for i, m in enumerate(mods):
            if (m.get("source"), m.get("destination")) == FILTER_ENV:
                return i + 1
        return None

    def _controls(self) -> dict:
        self.synth.load_json(self.state)
        return self.synth.get_controls()

    def _read(self, controls) -> dict[str, float]:
        out = {}
        for name in EDITABLE:
            if name != PEAK:
                out[name] = read_normalized(controls[name], name)
            elif self.env_slot is not None:
                depth = controls[f"modulation_{self.env_slot}_amount"].value()
                out[name] = out[CUTOFF] if abs(depth) < STILL else float(np.clip(out[CUTOFF] + depth, 0.0, 1.0))
        return out

    def values(self) -> dict[str, float]:
        """Exact 0-1 positions of the editable parameters in the current state."""
        return self._read(self._controls())

    def params(self) -> dict[str, dict]:
        """The editable parameters as an agent sees them: 0-1 position (`value`), the value with its unit
        (`text`), the allowed `range`, the text on the demo panel (`display`), and a description."""
        shaped = self.shaped()
        out = {}
        for name, value in self.values().items():
            text = unit_of(name).text(to_unit(name, value))
            out[name] = {
                "value": round(value, 3),
                "text": text,
                "range": range_text(name),
                "display": position_text(value) if name in shaped else text,
                "desc": f"{EDITABLE[name]} ({POSITIONS}; blends in between)" if name in shaped else EDITABLE[name],
            }
        return out

    def shaped(self) -> set[str]:
        """Wavetable position controls whose oscillator carries the shapes table in the current state."""
        preset = json.loads(self.state)
        return {f"osc_{osc}_wave_frame" for osc in OSCILLATORS if has_shapes(preset, osc)}

    def apply(self, edits: dict[str, float]) -> list[str]:
        """Set 0-1 positions of editable parameters; returns the names actually applied.

        The peak is a cutoff position of its own: setting it leaves the cutoff alone, and moving the cutoff leaves
        an existing peak where it was. While the two are equal (no movement), the peak follows the cutoff, so
        changing the cutoff alone never starts a sweep."""
        controls = self._controls()
        before = self._read(controls)
        applied = []
        for name, value in edits.items():
            if name in EDITABLE and name != PEAK:
                write_normalized(controls[name], name, value)
                applied.append(name)
        if self.env_slot is not None:
            depth = controls[f"modulation_{self.env_slot}_amount"]
            cutoff = read_normalized(controls[CUTOFF], CUTOFF)
            peak = None
            if PEAK in edits:
                peak = float(np.clip(float(edits[PEAK]), 0.0, 1.0))
                applied.append(PEAK)
            elif LEGACY_DEPTH in edits:
                depth.set(float(np.clip(2 * float(edits[LEGACY_DEPTH]) - 1, -1.0, 1.0)))
            elif abs(depth.value()) >= STILL and cutoff != before[CUTOFF]:
                peak = before[PEAK]
            if peak is not None:
                depth.set(0.0 if abs(peak - cutoff) < STILL else peak - cutoff)
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
