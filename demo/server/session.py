"""One in-memory editing session: revisions of a Vital state, an undo stack, and chat turns.

`VitalEnv` is the only source of truth: every revision stores the full preset JSON and, computed once when the
revision is made, its exact control values, display texts, descriptors, and rendered WAV.
"""

from __future__ import annotations

import io
import json
import random
import threading
import time
import zlib
from dataclasses import dataclass, field

import soundfile as sf

from synth_rl.features import describe
from synth_rl.procedural import init_patch, random_patch
from synth_rl.vital_env import CUTOFF, EDITABLE, PEAK, SR, VitalEnv, moved, preset_name
from synth_rl.wavetable import OSCILLATORS
from synth_rl.wavetable import frames as wavetable_frames

COALESCE_SECS = 2.0
CONTEXT_EVENTS = 12  # most recent turns and manual changes passed to the model
MOVED = 1e-6  # smaller differences in a normalized value are not a change


def group_of(name: str) -> str:
    if name.startswith("osc_1_"):
        return "osc1"
    if name.startswith("osc_2_"):
        return "osc2"
    if name.startswith("filter_1_"):
        return "filter"
    if name.startswith("env_1_"):
        return "env1"
    if name.startswith("env_2_"):
        return "env2"
    return name.split("_", 1)[0]  # chorus, reverb, delay, distortion, eq


def kind_of(name: str) -> str:
    if name.endswith("_on"):
        return "switch"
    if name.endswith("_unison_voices"):
        return "stepped"
    return "knob"


LABELS = {CUTOFF: "filter 1 cutoff", PEAK: "filter 1 cutoff at envelope 2 peak"}


def label_of(name: str) -> str:
    """Short name for the panel and the change rows: the prompt description without its explanation."""
    return LABELS.get(name) or EDITABLE.get(name, name).split(" (")[0]


@dataclass
class Revision:
    rev: int
    parent: int | None
    source: str  # "init" | "new_patch" | "manual" | "agent"
    summary: str
    state: str
    values: dict[str, float]
    displays: dict[str, str]
    descriptors: dict[str, float]
    wav: bytes
    name: str
    created: float = field(default_factory=time.time)
    manual_param: str = ""  # the single parameter a manual set moved, for coalescing
    wavetables: dict = field(default_factory=dict)  # per oscillator, the table's keyframes for the waveform display

    def json(self) -> dict:
        return {
            "rev": self.rev, "parent": self.parent, "source": self.source,
            "controls": [{"name": n, "label": label_of(n), "group": group_of(n), "kind": kind_of(n),
                          "value": self.values[n], "display": self.displays[n]} for n in EDITABLE if n in self.values],
            "descriptors": self.descriptors,
            # The query string changes with the audio itself. A revision number alone is not a stable name: numbers
            # restart with the server and a manual revision can be replaced in place, and a browser that cached the
            # old clip under the same URL would play a sound that no longer matches the knobs.
            "audio_url": f"/api/wav/{self.rev}.wav?v={zlib.crc32(self.wav):08x}",
            "preset_name": self.name,
            "wavetables": self.wavetables,
        }


@dataclass
class Turn:
    turn_id: str
    text: str
    from_rev: int
    to_rev: int
    changes: dict[str, tuple[str, str]]  # name -> (first from_display, last to_display)


def snapshot(env: VitalEnv, state: str) -> dict:
    """Exact values, display texts, descriptors, and WAV of one state."""
    env.load_state(state)
    values = env.values()
    displays = {name: p["display"] for name, p in env.params().items()}
    audio = env.render(env.state)
    buf = io.BytesIO()
    sf.write(buf, audio.T, SR, format="WAV", subtype="PCM_16")
    preset = json.loads(env.state)
    return {"state": env.state, "values": values, "displays": displays, "descriptors": describe(audio),
            "wav": buf.getvalue(), "name": preset_name(env.state) or "untitled",
            "wavetables": {f"osc_{osc}": wavetable_frames(preset, osc) for osc in OSCILLATORS}}


def changes_between(a: Revision, b: Revision) -> list[dict]:
    return [{"name": n, "from": a.values[n], "to": b.values[n], "from_display": a.displays[n],
             "to_display": b.displays[n]} for n in moved(a.values, b.values, MOVED)]


def delta(a: Revision, b: Revision) -> dict[str, float]:
    return {k: b.descriptors[k] - a.descriptors[k] for k in a.descriptors}


class Session:
    """Revisions by id (never reused, so audio URLs stay valid), a linear undo stack, and a cursor into it."""

    def __init__(self, seed: int = 7) -> None:
        self.lock = threading.RLock()
        self.env = VitalEnv()
        self.revisions: dict[int, Revision] = {}
        self.stack: list[int] = []
        self.cursor = -1
        self.turns: list[Turn] = []
        self.next_rev = 0
        self.next_turn = 1
        self.new_patch(seed=seed)

    # --- revisions -------------------------------------------------------------------------------------------

    def make_revision(self, snap: dict, parent: int | None, source: str, summary: str, manual_param: str = "") -> Revision:
        with self.lock:
            rev = Revision(self.next_rev, parent, source, summary, snap["state"], snap["values"], snap["displays"],
                           snap["descriptors"], snap["wav"], snap["name"], manual_param=manual_param,
                           wavetables=snap.get("wavetables", {}))
            self.next_rev += 1
            self.revisions[rev.rev] = rev
            return rev

    def push(self, rev: Revision, replace_current: bool = False) -> None:
        """Make `rev` current: drop the redo part of the stack, then append (or replace the current entry)."""
        with self.lock:
            del self.stack[self.cursor + 1:]
            if replace_current and self.stack:
                self.stack[-1] = rev.rev
            else:
                self.stack.append(rev.rev)
            self.cursor = len(self.stack) - 1

    @property
    def current(self) -> Revision:
        return self.revisions[self.stack[self.cursor]]

    def state_json(self) -> dict:
        with self.lock:
            return {
                "current": self.current.json(),
                "can_undo": self.cursor > 0,
                "can_redo": self.cursor < len(self.stack) - 1,
                "history": [{"rev": r, "source": self.revisions[r].source, "summary": self.revisions[r].summary}
                            for r in self.stack],
            }

    # --- actions ---------------------------------------------------------------------------------------------

    def new_patch(self, seed: int | None = None, init: bool = False) -> dict:
        with self.lock:
            if init:
                snap, source, summary = snapshot(self.env, init_patch(self.env)), "init", "init patch"
            else:
                seed = random.randrange(2**31) if seed is None else int(seed)
                state = random_patch(self.env, random.Random(seed))
                snap, source, summary = snapshot(self.env, state), "new_patch", f"new patch (seed {seed})"
            rev = self.make_revision(snap, None, source, summary)
            self.stack, self.cursor, self.turns = [rev.rev], 0, []
            return self.state_json()

    def set(self, edits: dict[str, float]) -> dict:
        unknown = [k for k in edits if k not in EDITABLE]
        if unknown:
            raise KeyError(f"not an editable parameter: {', '.join(unknown)}")
        with self.lock:
            cur = self.current
            self.env.load_state(cur.state)
            self.env.apply(edits)
            snap = snapshot(self.env, self.env.state)
            single = next(iter(edits)) if len(edits) == 1 else ""
            coalesce = (single and cur.source == "manual" and cur.manual_param == single
                        and time.time() - cur.created < COALESCE_SECS)
            parent = cur.parent if coalesce else cur.rev
            names = ", ".join(label_of(k) for k in edits) or "nothing"
            rev = self.make_revision(snap, parent, "manual", f"set {names}", manual_param=single)
            self.push(rev, replace_current=bool(coalesce))
            return self.state_json()

    def undo(self) -> dict:
        with self.lock:
            self.cursor = max(0, self.cursor - 1)
            return self.state_json()

    def redo(self) -> dict:
        with self.lock:
            self.cursor = min(len(self.stack) - 1, self.cursor + 1)
            return self.state_json()

    def ancestors(self, rev: int) -> set[int]:
        out = set()
        r: int | None = rev
        while r is not None and r not in out:
            out.add(r)
            r = self.revisions[r].parent
        return out

    def context_for(self, from_rev: int) -> list[str]:
        """What happened on the way to `from_rev`, oldest first: every earlier turn (the request and what the model
        changed) and every change the user made by hand. With the last turn alone, a model asked for "a sine pluck"
        two turns after "make it a pluck" and one after "too short!!" shortened the decay again in 6 of 6 replays;
        with the whole conversation, in none.

        A turn that started at from_rev itself is a sibling attempt (e.g. the unmuted run), not a predecessor, so
        a muted rerun gets the same context as the turn it repeats."""
        with self.lock:
            lineage = self.ancestors(from_rev)
            events: list[tuple[int, int, str]] = []
            for turn in self.turns:
                if turn.to_rev in lineage and turn.from_rev != from_rev:
                    events.append((turn.to_rev, len(events),
                                   f'The user asked: "{turn.text}". You changed: {_moves(turn.changes) or "nothing"}.'))
            for r in lineage:
                rev = self.revisions[r]
                if rev.source == "manual" and rev.parent is not None:
                    moved = {c["name"]: (c["from_display"], c["to_display"])
                             for c in changes_between(self.revisions[rev.parent], rev)}
                    if moved:
                        events.append((r, len(events), f"The user changed by hand: {_moves(moved)}."))
            return [text for _, _, text in sorted(events)][-CONTEXT_EVENTS:]

    def add_turn(self, turn: Turn) -> None:
        with self.lock:
            self.turns.append(turn)

    def new_turn_id(self) -> str:
        with self.lock:
            tid = f"t{self.next_turn}"
            self.next_turn += 1
            return tid


def _moves(changes: dict[str, tuple[str, str]]) -> str:
    return ", ".join(f"{n} {a or '?'} -> {b or '?'}" for n, (a, b) in changes.items())


def with_context(events: list[str], text: str) -> str:
    """The instruction sent to a model: the new message, after the conversation that led to the current sound."""
    if not events:
        return text
    lines = "\n".join(f"{i}. {e}" for i, e in enumerate(events, 1))
    return (f"The conversation so far:\n{lines}\n"
            f"Earlier requests still apply unless the new one contradicts them.\nNow: {text}")
