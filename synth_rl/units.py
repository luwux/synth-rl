"""Units of the editable parameters: what an agent reads and writes.

Vital keeps each control as a raw number on its own scale (a cutoff as a MIDI note number, an envelope time as the
fourth root of seconds), and vita adds a 0-1 position on that scale. Neither tells a model what the value is, so
prompts list every value in Hz, seconds, dB, percent, or voices together with its allowed range, and replies are
read in the same units. Inside the package, edits stay 0-1 positions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

MIDDLE_C_HZ = 261.63  # the note every render plays


@dataclass(frozen=True)
class Unit:
    unit: str  # "Hz", "s", "dB", "%", "voices"; "" for a switch
    from_raw: Callable[[float], float]  # Vital's raw control value -> value in `unit`
    to_raw: Callable[[float], float]

    def text(self, value: float) -> str:
        """The value as shown to agents and on the demo panel: "1240 Hz", "0.48 s", "on"."""
        if not self.unit:
            return "on" if value >= 0.5 else "off"
        if self.unit == "voices":
            return f"{round(value)} voice{'' if round(value) == 1 else 's'}"
        return f"{number_text(value)} {self.unit}"

    def read(self, value) -> float | None:
        """A value from a reply, in this unit: a number, or text such as "800 Hz", "1.2 kHz", "300 ms", "on"."""
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value) if math.isfinite(value) else None
        if not isinstance(value, str):
            return None
        word = value.strip().lower()
        if word in WORDS:
            return WORDS[word]
        m = re.fullmatch(r"([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*([a-z%]*)", word)
        if not m:
            return None
        scale = {("Hz", "khz"): 1000.0, ("s", "ms"): 0.001}.get((self.unit, m.group(2)), 1.0)
        return float(m.group(1)) * scale


WORDS = {"on": 1.0, "true": 1.0, "yes": 1.0, "off": 0.0, "false": 0.0, "no": 0.0}


def number_text(value: float) -> str:
    """Three significant digits, at most four decimals, no exponent: 1240, 56.7, 0.48, 0.0005, 0."""
    if abs(value) < 5e-5:
        return "0"
    decimals = min(4, 2 - math.floor(math.log10(abs(value))))
    text = f"{round(value, decimals):.{max(0, decimals)}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


PERCENT = Unit("%", lambda r: r * 100, lambda v: v / 100)
# Vital shows the square of these controls: level as a gain from 0 to 1, detune from 0 to 100 %.
LEVEL = Unit("%", lambda r: r * r * 100, lambda v: math.sqrt(max(v, 0.0) / 100))
DETUNE = Unit("%", lambda r: r * r, lambda v: math.sqrt(max(v, 0.0)))
TIME = Unit("s", lambda r: r**4, lambda v: max(v, 0.0) ** 0.25)
REVERB_TIME = Unit("s", lambda r: 2.0**r, lambda v: math.log2(max(v, 1e-9)))
CUTOFF = Unit("Hz", lambda r: 440.0 * 2 ** ((r - 69) / 12), lambda v: 69 + 12 * math.log2(max(v, 1e-9) / 440.0))
DB = Unit("dB", lambda r: r, lambda v: v)
VOICES = Unit("voices", lambda r: float(round(r)), lambda v: float(round(v)))
SWITCH = Unit("", lambda r: float(r >= 0.5), lambda v: float(v >= 0.5))
POSITION = Unit("%", lambda r: r / 2.56, lambda v: v * 2.56)  # along the 257 frames of a wavetable

PATTERNS = [  # first match wins
    (r"_on$", SWITCH),
    (r"_unison_voices$", VOICES),
    (r"_unison_detune$", DETUNE),
    (r"^osc_\d_level$", LEVEL),
    (r"_wave_frame$", POSITION),
    (r"cutoff", CUTOFF),
    (r"^env_\d_(attack|decay|release)$", TIME),
    (r"^reverb_decay_time$", REVERB_TIME),
    (r"(_drive|_gain)$", DB),
    (r"", PERCENT),
]


def unit_of(name: str) -> Unit:
    return next(unit for pattern, unit in PATTERNS if re.search(pattern, name))
