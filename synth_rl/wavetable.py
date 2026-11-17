"""The wavetable of generated patches: five basic shapes spread along the position knob.

Vital's init preset carries a table with a single frame (a saw), on which the wavetable position knobs do nothing.
Generated patches use this table instead, so position 0 is a sine, 0.25 a triangle, 0.5 the init saw, 0.75 a square
and 1 a narrow pulse, with Vital morphing between neighbours.

The shapes share the saw's RMS level and the phase of its fundamental, and neighbours cross-fade in the time
domain. So the knob changes the harmonics without acting as a volume control (the level stays within about 1 dB
along the whole range), and the saw frame renders exactly as the init preset does. Vital's spectral morph, the init
table's setting, produces in-between frames with higher peaks, and since the table is normalized as a whole that
would turn every shape down.
"""

from __future__ import annotations

import base64

import numpy as np

SHAPES = ("sine", "triangle", "saw", "square", "pulse")
TABLE_NAME = "Shapes"
MAX_FRAME = 256  # top of Vital's wave frame controls
SAW_FRAME = MAX_FRAME * SHAPES.index("saw") // (len(SHAPES) - 1)
POSITIONS = ", ".join(f"{100 * i / (len(SHAPES) - 1):g} {s}" for i, s in enumerate(SHAPES))  # in percent
OSCILLATORS = (1, 2)  # the oscillators agents can edit


def _encode(wave: np.ndarray) -> str:
    return base64.b64encode(wave.astype(np.float32).tobytes()).decode()


def _keyframes(saw_b64: str) -> list[dict]:
    """Keyframes built around the init table's own saw, which is kept byte for byte."""
    saw = np.frombuffer(base64.b64decode(saw_b64), dtype=np.float32)
    phase = 2 * np.pi * (np.arange(len(saw)) + 0.5) / len(saw)
    waves = {
        "sine": -np.sin(phase),
        "triangle": -2 / np.pi * np.arcsin(np.sin(phase)),
        "square": -np.sign(np.sin(phase)),
        "pulse": np.where(np.cos(phase - 1.5 * np.pi) > np.cos(np.pi / 4), 1.0, -1 / 3),  # high for a quarter cycle
    }
    level = np.sqrt((saw.astype(np.float64) ** 2).mean())
    frames = []
    for i, shape in enumerate(SHAPES):
        wave = waves.get(shape)
        data = saw_b64 if wave is None else _encode(wave * level / np.sqrt((wave**2).mean()))
        frames.append({"position": MAX_FRAME * i // (len(SHAPES) - 1), "wave_data": data})
    return frames


def use_shapes(preset: dict) -> None:
    """Give the editable oscillators of an init-preset dict the shapes table, positioned on the saw."""
    for osc in OSCILLATORS:
        table = preset["settings"]["wavetables"][osc - 1]
        component = table["groups"][0]["components"][0]
        component["keyframes"] = _keyframes(component["keyframes"][0]["wave_data"])
        component["interpolation"] = 0  # cross-fade in the time domain
        table["name"] = TABLE_NAME
        preset["settings"][f"osc_{osc}_wave_frame"] = float(SAW_FRAME)


def has_shapes(preset: dict, osc: int) -> bool:
    return preset["settings"]["wavetables"][osc - 1].get("name") == TABLE_NAME


def frames(preset: dict, osc: int, points: int = 128) -> dict:
    """The keyframes of an oscillator's table for drawing: {"positions": [0-1, ...], "frames": [[samples], ...]},
    each frame thinned to `points` samples and the whole table scaled to a peak of 1. With this module's table (and
    any table that cross-fades in the time domain) the wave at a position is the straight blend of the two keyframes
    around it."""
    component = preset["settings"]["wavetables"][osc - 1]["groups"][0]["components"][0]
    keys = sorted(component.get("keyframes", []), key=lambda k: k["position"])
    waves = [np.frombuffer(base64.b64decode(k["wave_data"]), dtype=np.float32) for k in keys]
    peak = max([float(np.abs(w).max()) for w in waves] + [1e-9])
    thin = [w[((np.arange(points) + 0.5) * len(w) / points).astype(int)] / peak for w in waves]
    return {"positions": [round(k["position"] / MAX_FRAME, 4) for k in keys],
            "frames": [[round(float(x), 3) for x in w] for w in thin]}


def position_text(normalized: float) -> str:
    """Where a normalized position sits in the table: "saw", or "saw to square, 30%"."""
    x = float(np.clip(normalized, 0.0, 1.0)) * (len(SHAPES) - 1)
    i = min(int(x), len(SHAPES) - 2)
    mix = round((x - i) * 100)
    if mix in (0, 100):
        return SHAPES[i + mix // 100]
    return f"{SHAPES[i]} to {SHAPES[i + 1]}, {mix}%"
