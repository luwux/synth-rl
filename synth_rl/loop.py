"""Step-by-step episodes: the agent sets parameters, hears the result, and decides the next step.

One-shot episodes (`cli.run_episode`: one listing, one clip, one JSON reply) stay the diagnostic setting, because
nothing in them depends on what the model chooses to do. This module is the loop for agents and for RL on
trajectories, kept to three pieces:

- one tool, `set`, taking any number of parameters, declared once from the parameter table (`tool_schema`);
- `Episode`, a reset/step interface over a task: every step applies one reply, returns what changed together with
  the clip of the new sound, and records the task reward before and after;
- `rollout`, the loop itself: ask the agent, step, repeat until it says done, changes nothing, or runs out of steps.

The audio of every result is always given; the agent does not have to ask for it. So a muted run (silent clips, same
messages otherwise) differs from a normal one in the audio alone.

Messages are OpenAI chat format with two kinds of content part, {"type": "text"} and {"type": "audio"}. A step is an
assistant message holding the JSON action, then a user message with the result text and the new clip. That form
needs no tool-calling support from the model's chat template, so an API model and a locally loaded one read the same
conversation; `as_tool_calls` rewrites a record into assistant `tool_calls` / `tool` messages for trainers that expect
them (tool messages carry text only in that format, so the clip stays in a user message).
"""

from __future__ import annotations

import json

import numpy as np

from .agents import first_json, hide_values, prompt_text, read_reply
from .tasks import Task, changed_params, score_against_audio, score_state
from .units import unit_of
from .vital_env import EDITABLE, VitalEnv, moved, range_text

def _system(act: str, step: str, stop: str, again: str) -> str:
    return f"""You are a sound designer operating the Vital wavetable synthesizer, one step at a time.
You receive the parameters you can set, each with its current value, its unit and allowed range, and what it does,
plus audio of the current sound (one held middle-C note, 261.6 Hz: 1.5 s held, 3 s total).
{act}
Make the whole change in one {step} when you can: one {step} may set any number of parameters. After each {step} you get
the list of what changed and audio of the new sound. Listen to it. If it does what was asked, {stop}.
{again} only to correct something you hear is wrong or still missing; do not push a change further than
was asked.
Give each value as a number in the unit listed for that parameter (Hz, s, dB, %, voices); switches take "on" or
"off". Values outside the range are clipped to it. Only use parameter names from the list, and change as few
parameters as needed."""


# Stopping is the default: with "either take another step or, when the sound is what was asked for, reply done" and
# nothing after the clip, the model kept pushing the same change ("make it warmer!": cutoff 800, 600, 450, 300 Hz).
STEP_SYSTEM = _system('Reply with JSON only. To change the sound: {"set": {"<parameter>": <new value>, ...}, '
                      '"reason": "<one sentence>"}.', "step", 'reply {"done": true}', "Take another step")
RESULT_END = 'If this does what was asked, reply {"done": true}.'  # after the clip of every result
# The same conversation with native tool calls (`as_tool_calls`).
TOOL_SYSTEM = _system("To change the sound, call the `set` tool.", "call", 'reply "done" without calling the tool',
                      "Call it again")
TOOL_END = 'If this does what was asked, reply "done" without calling the tool.'


def tool_schema() -> dict:
    """The `set` tool in OpenAI function format: every editable parameter with its unit, range, and description."""
    properties = {}
    for name, desc in EDITABLE.items():
        unit = unit_of(name).unit
        properties[name] = ({"type": "string", "enum": ["on", "off"], "description": desc} if not unit else
                            {"type": "number", "description": f"{desc}. In {unit}, {range_text(name)}."})
    return {"type": "function", "function": {
        "name": "set",
        "description": "Set any number of synthesizer parameters. Returns what changed and audio of the new sound.",
        "parameters": {"type": "object", "properties": properties, "additionalProperties": False}}}


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


class Episode:
    """One task, stepped. `messages` is the conversation so far (audio parts hold arrays); `steps` the records."""

    def __init__(self, env: VitalEnv, task: Task, blind: bool = False, target_audio: np.ndarray | None = None,
                 renders: int = 1):
        self.env, self.task, self.blind, self.renders = env, task, blind, renders
        self.state = task.state
        self.target_audio = env.render(task.target_state) if task.target_state else target_audio
        self.steps: list[dict] = []
        self.stopped = ""  # why the episode ended: "done", "no change", "unreadable", or "max steps"
        env.load_state(task.state)
        params = env.params()
        content = [_text(prompt_text(hide_values(params) if blind else params, task.instruction)),
                   {"type": "audio", "clip": "start", "audio": env.render(task.state)}]
        if self.target_audio is not None:
            content += [_text("Second clip: target sound."), {"type": "audio", "clip": "target", "audio": self.target_audio}]
        self.messages = [{"role": "system", "content": STEP_SYSTEM}, {"role": "user", "content": content}]
        self.reward = self._reward(task.state)

    def _result(self, state: str) -> dict:
        task = self.task
        if task.kind == "edit" and not task.spec_id:  # a free-form instruction has no reward
            return {"n_changed": len(changed_params(self.env, task.state, state))}
        if task.kind != "edit" and not task.target_state:
            return score_against_audio(self.env, task.state, state, self.target_audio)
        return score_state(self.env, task, state, self.renders)

    def _reward(self, state: str) -> float | None:
        return self._result(state).get("reward")

    def step(self, reply: str) -> dict:
        """Apply one reply of the agent and append the result turn. Returns the step's record."""
        number = len(self.steps) + 1
        self.messages.append({"role": "assistant", "content": reply})
        self.env.load_state(self.state)
        params, before = self.env.params(), self.env.values()
        parsed = read_reply(reply, params)
        done = (first_json(reply) or {}).get("done") is True
        self.env.apply(parsed.edits)
        names = moved(before, self.env.values(), 1e-6)
        shown = self.env.params()
        changes = {n: {"from": params[n]["display"], "to": shown[n]["display"]} for n in names}
        record = {"step": number, "reply": reply, "edits": parsed.edits, "changes": changes, "clipped": parsed.clipped,
                  "valid": bool(parsed.readable) or done, "reward_before": self.reward}
        if names:
            self.state = self.env.state
            self.reward = self._reward(self.state)
            listing = ", ".join(f"{n} -> {c['to']}" if self.blind else f"{n} {c['from']} -> {c['to']}" for n, c in changes.items())
            clipped = f" Clipped to the allowed range: {', '.join(parsed.clipped)}." if parsed.clipped else ""
            self.messages.append({"role": "user", "content": [
                _text(f"Step {number} applied: {listing}.{clipped} New sound:"),
                {"type": "audio", "clip": f"step{number}", "audio": self.env.render(self.state)},
                *([_text(RESULT_END)] if RESULT_END else [])]})
            if done:
                self.stopped = "done"
        else:
            self.stopped = "done" if done else "no change" if parsed.readable else "unreadable"
        record["reward"] = self.reward
        self.steps.append(record)
        return record

    @property
    def done(self) -> bool:
        return bool(self.stopped)

    def record(self) -> dict:
        """The finished episode: final score, per-step records, and the conversation with clips named, not stored
        (start, target, step1, ...; each is the render of the state after that step, rebuilt from `turns`)."""
        messages = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else
                     [{"type": "audio", "clip": p["clip"]} if p["type"] == "audio" else p for p in m["content"]]}
                    for m in self.messages]
        return {**self._result(self.state), "stopped": self.stopped or "max steps", "n_steps": len(self.steps),
                "steps": self.steps, "turns": [{"edits": s["edits"]} for s in self.steps], "messages": messages}


def rollout(agent, env: VitalEnv, task: Task, max_steps: int = 6, blind: bool = False,
            target_audio: np.ndarray | None = None, renders: int = 1) -> dict:
    """Run one episode: `agent.complete(messages)` returns the next reply."""
    episode = Episode(env, task, blind, target_audio, renders)
    while not episode.done and len(episode.steps) < max_steps:
        episode.step(agent.complete(episode.messages))
    return episode.record()


def as_tool_calls(record: dict) -> tuple[list[dict], list[dict]]:
    """A record's conversation with every step as a native tool call: an assistant message with `tool_calls`, a
    `tool` message with the result text, and a user message with the clip. Returns (messages, tools)."""
    out, pending = [], None
    for i, message in enumerate(record["messages"]):
        if message["role"] == "system":
            out.append({"role": "system", "content": TOOL_SYSTEM if message["content"] == STEP_SYSTEM else message["content"]})
        elif message["role"] == "assistant":
            data = first_json(message["content"]) or {}
            edits = data.get("set", data.get("edits"))
            if isinstance(edits, dict):
                pending = f"call_{i}"
                out.append({"role": "assistant", "content": str(data.get("reason", "")), "tool_calls": [{
                    "id": pending, "type": "function", "function": {"name": "set", "arguments": json.dumps(edits)}}]})
            else:
                out.append({"role": "assistant", "content": str(data.get("reason", "")) or "done"})
        elif message["role"] == "user" and pending:
            parts = message["content"]
            out.append({"role": "tool", "tool_call_id": pending, "content": parts[0]["text"]})
            out.append({"role": "user", "content": [p for p in parts if p["type"] == "audio"] + [_text(TOOL_END)]})
            pending = None
        else:
            out.append(message)
    if pending:  # the last call changed nothing, which ended the episode
        out.append({"role": "tool", "tool_call_id": pending, "content": "Nothing changed."})
    return out, [tool_schema()]
