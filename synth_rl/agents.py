"""Agents that propose parameter edits: a keyword rule baseline and a Gemini audio-LLM agent."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request

import numpy as np
import soundfile as sf

from .vital_env import SR

SYSTEM = """You are a sound designer operating the Vital wavetable synthesizer.
You receive the current parameters (normalized 0-1 values plus Vital's display text) and audio of the current sound
(one held middle-C note, 1.5 s held, 3 s total). Listen to the audio before deciding.
Reply with JSON only: {"edits": {"<param>": <new normalized value>, ...}, "reason": "<one sentence>"}.
Only use parameter names from the list. Change as few parameters as needed. Switches are 0 (off) or 1 (on)."""


def prompt_text(params: dict, instruction: str) -> str:
    """User-turn text shared by evaluation agents and exported RL prompts. Parameters whose value is None
    (see `hide_values`) are listed without their current setting."""
    hidden = any(v["value"] is None for v in params.values())
    listing = "\n".join(f"- {k} — {v['desc']}" if v["value"] is None else f"- {k}: {v['value']} ({v['display']}) — {v['desc']}"
                         for k, v in params.items())
    note = "\n(Current values are hidden: judge them by listening.)" if hidden else ""
    return f"Parameters:\n{listing}{note}\n\nInstruction: {instruction}\n\nFirst clip: current sound."


def hide_values(params: dict) -> dict:
    """Blind-knob variant: the agent sees parameter names but must infer their settings from the audio."""
    return {k: {**v, "value": None, "display": ""} for k, v in params.items()}


def wav_b64(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio.T if audio.ndim == 2 else audio, SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def first_json(text: str) -> dict | None:
    """The first JSON object in `text`; models sometimes append extra text or more objects."""
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def parse_edits(text: str) -> dict[str, float]:
    data = first_json(text)
    if data is None:
        return {}
    edits = data.get("edits", data)
    out = {}
    for k, v in edits.items() if isinstance(edits, dict) else []:
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


class RuleAgent:
    """Keyword baseline: fixed parameter moves per instruction family. Does not listen."""

    name = "rule"

    # First match wins, so more specific patterns come first.
    RULES = [
        (r"narrow|mono|collapse", {"chorus_dry_wet": -0.5, "osc_1_stereo_spread": -0.5}),
        (r"wider|spread|stereo width", {"chorus_on": 1.0, "chorus_dry_wet": +0.3, "osc_1_stereo_spread": +0.3}),
        (r"drier|dry|close|shorten the tail", {"reverb_dry_wet": -0.4, "delay_dry_wet": -0.4}),
        (r"space|reverb|room|tail", {"reverb_on": 1.0, "reverb_dry_wet": +0.3, "reverb_decay_time": +0.2}),
        (r"bright|top end|sparkle|less muffled", {"filter_1_cutoff": +0.2, "eq_high_gain": +0.15}),
        (r"dark|tame|warm|harsh|muffled", {"filter_1_cutoff": -0.2, "eq_high_gain": -0.15}),
        (r"pluck|shorter", {"env_1_sustain": -0.6, "env_1_decay": -0.15}),
        (r"sustain|pad|hold", {"env_1_sustain": +0.6}),
        (r"attack|fade|swell", {"env_1_attack": +0.3}),
    ]

    def propose(self, instruction: str, params: dict, audio: np.ndarray, target: np.ndarray | None = None) -> dict:
        text = instruction.lower()
        for pattern, moves in self.RULES:
            if re.search(pattern, text):
                edits = {}
                for name, delta in moves.items():
                    if name.endswith("_on"):
                        edits[name] = delta
                    else:  # with hidden values the rule assumes the knob is centred
                        current = params[name]["value"]
                        edits[name] = (0.5 if current is None else current) + delta
                    if name.startswith("eq_"):
                        edits["eq_on"] = 1.0
                return {"edits": edits, "reason": f"rule: {pattern}"}
        return {"edits": {}, "reason": "no rule matched"}


class GeminiAgent:
    """Gemini via the REST generateContent API with inline WAV audio."""

    def __init__(self, model: str = "gemini-3.8-flash", mute_audio: bool = False, temperature: float = 0.7):
        self.model = model
        self.mute_audio = mute_audio
        self.temperature = temperature
        self.key = os.environ.get("GEMINI_API_KEY", "")
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.name = f"{model}{'+mute' if mute_audio else ''}"

    def _audio_part(self, audio: np.ndarray) -> dict:
        if self.mute_audio:
            audio = np.zeros_like(audio)
        return {"inlineData": {"mimeType": "audio/wav", "data": wav_b64(audio)}}

    def propose(self, instruction: str, params: dict, audio: np.ndarray, target: np.ndarray | None = None) -> dict:
        parts = [{"text": prompt_text(params, instruction)}, self._audio_part(audio)]
        if target is not None:
            parts.append({"text": "Second clip: target sound."})
            parts.append(self._audio_part(target))
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": self.temperature, "responseMimeType": "application/json"},
        }
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.key},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        reason = (first_json(text) or {}).get("reason", "")
        return {"edits": parse_edits(text), "reason": str(reason), "raw": text}
