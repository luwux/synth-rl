"""Agents that propose parameter edits: a keyword rule baseline, Gemini, and OpenAI-compatible chat endpoints."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from typing import NamedTuple

import numpy as np
import soundfile as sf

from .units import unit_of
from .vital_env import EDITABLE, LEGACY_DEPTH, SR, from_unit, to_unit, unit_range

SYSTEM = """You are a sound designer operating the Vital wavetable synthesizer.
You receive the parameters you can set, each with its current value, its unit and allowed range, and what it does,
plus audio of the current sound (one held middle-C note, 261.6 Hz: 1.5 s held, 3 s total). Listen to the audio
before deciding.
Reply with JSON only: {"edits": {"<parameter>": <new value>, ...}, "reason": "<one sentence>"}.
Give each value as a number in the unit listed for that parameter (Hz, s, dB, %, voices); switches take "on" or
"off". Values outside the range are clipped to it. Only use parameter names from the list, and change as few
parameters as needed."""


def prompt_text(params: dict, instruction: str) -> str:
    """User-turn text shared by evaluation agents and exported RL prompts: one line per parameter with its value
    in units, its range, and its description. Parameters whose value is hidden (see `hide_values`) show "?"."""
    hidden = any(v["value"] is None for v in params.values())
    listing = "\n".join(f"- {k}: {'?' if v['value'] is None else v['text']} ({v['range']}) — {v['desc']}"
                         for k, v in params.items())
    note = "\n(Current values are hidden: judge them by listening.)" if hidden else ""
    return f"Parameters:\n{listing}{note}\n\nInstruction: {instruction}\n\nFirst clip: current sound."


def hide_values(params: dict) -> dict:
    """Blind-knob variant: the agent sees parameter names, units, and ranges but must infer the settings from
    the audio."""
    return {k: {**v, "value": None, "text": "", "display": ""} for k, v in params.items()}


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


class Reply(NamedTuple):
    edits: dict[str, float]  # 0-1 positions to apply
    clipped: list[str]  # names whose value lay outside its range
    readable: int  # edits that could be read, including ones that restate the current value


def read_reply(text: str, params: dict | None = None) -> Reply:
    """The edits of a reply as 0-1 positions.

    Values are read in each parameter's unit. With `params` (the listing the reply answered), a value that reads
    the same as the current one is dropped, so a reply that restates the whole patch changes only what it changed.
    The edits sit under "edits" (one-shot replies) or "set" (steps of `loop`). Unknown names and unreadable values
    are dropped; the pre-2026-11-16 depth name passes through as a position."""
    data = first_json(text)
    edits = data.get("edits", data.get("set", data)) if data is not None else None
    out, clipped, readable = {}, [], 0
    for name, given in edits.items() if isinstance(edits, dict) else []:
        name = str(name)
        if name == LEGACY_DEPTH:
            try:
                out[name] = float(given)
                readable += 1
            except (TypeError, ValueError):
                pass
            continue
        value = unit_of(name).read(given) if name in EDITABLE else None
        if value is None:
            continue
        readable += 1
        position = from_unit(name, value)
        shown = (params or {}).get(name, {}).get("text")
        if shown and unit_of(name).text(to_unit(name, position)) == shown:
            continue
        low, high = unit_range(name)  # the listed range is rounded, so allow that much
        if unit_of(name).unit and not low - 0.005 * abs(low) - 1e-4 <= value <= high + 0.005 * abs(high) + 1e-4:
            clipped.append(name)
        out[name] = position
    return Reply(out, clipped, readable)


def parse_edits(text: str, params: dict | None = None) -> dict[str, float] | None:
    """The edits to apply, or None when the reply holds no readable edit (a format failure). A reply that only
    restates current values is valid and gives {}."""
    reply = read_reply(text, params)
    return reply.edits if reply.readable else None


def reply_text(edits: dict[str, float], reason: str = "") -> str:
    """The reply that asks for these 0-1 positions, written as an agent writes it (values with units)."""
    return json.dumps({"edits": {k: unit_of(k).text(to_unit(k, v)) for k, v in edits.items() if k in EDITABLE},
                       "reason": reason})


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
        reply = read_reply(text, params)
        return {"edits": reply.edits, "clipped": reply.clipped, "valid": bool(reply.readable), "reason": str(reason),
                "raw": text}


OPENROUTER_URL = "https://openrouter.ai/api/v1"


class OpenAIChatAgent:
    """Any OpenAI-compatible chat-completions endpoint that accepts `input_audio` parts: OpenRouter by default,
    or a local server such as llama-server (`base_url="http://127.0.0.1:8080/v1"`, no key needed).

    `options` is merged into the request body; the default turns reasoning off and caps the reply, which keeps
    a three-line JSON answer from costing a long hidden chain of thought."""

    DEFAULT_OPTIONS = {"reasoning": {"enabled": False}, "max_tokens": 1024}

    def __init__(self, model: str, base_url: str = OPENROUTER_URL, key_env: str = "OPENROUTER_API_KEY",
                 mute_audio: bool = False, temperature: float = 0.7, options: dict | None = None,
                 timeout: float = 120, label: str = ""):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.mute_audio = mute_audio
        self.temperature = temperature
        self.options = dict(self.DEFAULT_OPTIONS if options is None else options)
        self.timeout = timeout
        self.key = os.environ.get(key_env, "") if key_env else ""
        if self.base_url == OPENROUTER_URL and not self.key:
            raise RuntimeError(f"{key_env} is not set")
        self.name = f"{label or model}{'+mute' if mute_audio else ''}"
        self.last_usage: dict = {}

    def _audio_part(self, audio: np.ndarray) -> dict:
        if self.mute_audio:
            audio = np.zeros_like(audio)
        return {"type": "input_audio", "input_audio": {"data": wav_b64(audio), "format": "wav"}}

    def complete(self, messages: list[dict]) -> str:
        """The model's reply to a conversation. Content parts {"type": "audio", "audio": <array>} are sent as WAV
        (as silence when this agent is muted)."""
        sent = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else
                 [self._audio_part(p["audio"]) if p["type"] == "audio" else p for p in m["content"]]} for m in messages]
        body = {"model": self.model, "temperature": self.temperature, "messages": sent, **self.options}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:  # the body names the problem (credits, model id); it never has the key
            detail = e.read()[:300].decode("utf-8", "replace")
            raise RuntimeError(f"{self.model}: HTTP {e.code} {detail}") from None
        if "choices" not in data:
            raise RuntimeError(f"{self.model}: no choices in reply: {json.dumps(data)[:300]}")
        self.last_usage = data.get("usage") or {}
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        if not text.strip() or (choice.get("finish_reason") == "length" and first_json(text) is None):
            raise RuntimeError(f"{self.model}: reply cut off or empty (finish_reason {choice.get('finish_reason')})")
        return text

    def propose(self, instruction: str, params: dict, audio: np.ndarray, target: np.ndarray | None = None) -> dict:
        content = [{"type": "text", "text": prompt_text(params, instruction)}, {"type": "audio", "audio": audio}]
        if target is not None:
            content.append({"type": "text", "text": "Second clip: target sound."})
            content.append({"type": "audio", "audio": target})
        text = self.complete([{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}])
        reason = (first_json(text) or {}).get("reason", "")
        reply = read_reply(text, params)
        return {"edits": reply.edits, "clipped": reply.clipped, "valid": bool(reply.readable), "reason": str(reason),
                "raw": text}
