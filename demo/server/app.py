"""FastAPI server for the synth-rl demo (contract: demo/DESIGN.md, "Server").

    uvicorn demo.server.app:app --host 127.0.0.1 --port 8765      # from the repo root, with .venv

Optional: SYNTH_RL_LOCAL_AGENT="openai:http://127.0.0.1:8080/v1::<model>" lists a local OpenAI-compatible
endpoint in /api/agents (available only if it answers; the server never starts one).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from synth_rl.agents import first_json
from synth_rl.cli import make_agent, run_episode
from synth_rl.loop import Episode
from synth_rl.tasks import Task
from synth_rl.vital_env import VitalEnv, preset_file
from synth_rl.vital_env import moved as moved_controls

from .session import MOVED, Session, Turn, delta, label_of, snapshot, with_context

DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
OPENROUTER_MODEL = "openrouter:qwen/qwen3.8-omni-flash"
CHAT_PREFIXES = ("openrouter:", "openai:")  # plus "rule"; never a local Qwen folder (would load a large model)
# Appended to a chat message for agents that hold a conversation. Without it a greeting or a question got edits as
# its answer ("um hi what si this": 4 steps, 8 controls moved).
CHAT_NOTE = ('\n(If this message does not ask for a change to the sound, such as a greeting or a question, change '
             'nothing and answer it instead: reply {"done": true, "reason": "<your answer in one or two sentences>"}.)')

app = FastAPI(title="synth-rl demo")
# The page is served from this server; CORS only matters for a Vite dev server on another local port.
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["*"], allow_headers=["*"])
session = Session()
chat_lock = asyncio.Lock()  # one chat turn at a time


class NewPatchBody(BaseModel):
    seed: int | None = None
    init: bool = False


class SetBody(BaseModel):
    edits: dict[str, float]


class RevBody(BaseModel):
    rev: int


class ChatBody(BaseModel):
    text: str
    agent: str = "rule"
    rounds: int = Field(4, ge=1, le=8)  # the most steps the agent may take in this turn
    mute: bool = False
    from_rev: int | None = None


def revision_or_404(rev: int):
    with session.lock:
        if rev not in session.revisions:
            raise HTTPException(404, f"no revision {rev}")
        return session.revisions[rev]


# --- state and editing -----------------------------------------------------------------------------------------

@app.get("/api/state")
def get_state():
    return session.state_json()


@app.post("/api/new_patch")
def new_patch(body: NewPatchBody | None = None):
    body = body or NewPatchBody()
    return session.new_patch(seed=body.seed, init=body.init)


@app.post("/api/set")
def set_params(body: SetBody):
    try:
        return session.set(body.edits)
    except KeyError as e:
        raise HTTPException(400, str(e.args[0])) from None


@app.post("/api/undo")
def undo():
    return session.undo()


@app.post("/api/redo")
def redo():
    return session.redo()


@app.get("/api/wav/{rev:int}.wav")
def audio(rev: int):
    r = revision_or_404(rev)
    # Revision numbers are never reused, so a revision's audio never changes.
    return Response(r.wav, media_type="audio/wav", headers={"Cache-Control": "no-store"})


def preset_filename(r) -> str:
    return f"{re.sub(r'[^A-Za-z0-9._-]+', '-', r.name).strip('-') or 'patch'}-rev{r.rev}.vital"


@app.get("/api/preset/{rev:int}.vital")
def preset(rev: int):
    r = revision_or_404(rev)
    return Response(preset_file(r.state), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{preset_filename(r)}"'})


@app.post("/api/open_in_vital")
def open_in_vital(body: RevBody):
    r = revision_or_404(body.rev)
    folder = Path(tempfile.gettempdir()) / "synth-rl-demo"
    folder.mkdir(exist_ok=True)
    path = folder / preset_filename(r)
    path.write_text(preset_file(r.state))
    try:
        # `open -a Vital <file>` is ignored by Vital; the file has to be a launch argument, so each click opens
        # a new Vital instance with the preset loaded.
        proc = subprocess.run(["open", "-n", "-a", "Vital", "--args", str(path)], capture_output=True, text=True,
                              timeout=15)
    except (OSError, subprocess.TimeoutExpired) as e:
        return JSONResponse({"ok": False, "message": f"could not run open: {e}"}, status_code=500)
    if proc.returncode != 0:
        return JSONResponse({"ok": False, "message": (proc.stderr or proc.stdout).strip()[:300]}, status_code=500)
    return {"ok": True}


# --- agents ----------------------------------------------------------------------------------------------------

def local_endpoint_up(agent_id: str) -> tuple[bool, str]:
    url = agent_id.removeprefix("openai:").partition("::")[0].rstrip("/")
    try:
        with urllib.request.urlopen(f"{url}/models", timeout=0.5):
            return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"no server answering at {url} ({type(e).__name__}); the demo never starts one"


@app.get("/api/agents")
def agents():
    has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    out = [
        {"id": OPENROUTER_MODEL, "label": "Qwen3.8-Omni-Flash (OpenRouter)", "listens": True, "available": has_key,
         "why": "" if has_key else "OPENROUTER_API_KEY is not set in the server's environment"},
        {"id": OPENROUTER_MODEL + "+think", "label": "Qwen3.8-Omni-Flash, reasoning on (slower)", "listens": True,
         "available": has_key, "why": "" if has_key else "OPENROUTER_API_KEY is not set in the server's environment"},
        {"id": "rule", "label": "Keyword rule (never listens)", "listens": False, "available": True, "why": ""},
    ]
    local = os.environ.get("SYNTH_RL_LOCAL_AGENT", "")
    if local.startswith("openai:"):
        up, why = local_endpoint_up(local)
        model = local.partition("::")[2] or "local model"
        out.append({"id": local, "label": f"{model} (local endpoint)", "listens": True, "available": up, "why": why})
    out.append({"id": "gemini-3.8-flash", "label": "Gemini 3.8 Flash", "listens": True, "available": False,
                "why": "API credits depleted"})
    return out


# --- chat ------------------------------------------------------------------------------------------------------

def run_round(agent, base_state: str, instruction: str) -> tuple[dict, dict]:
    """One one-shot reply on a fresh env (agents that cannot hold a conversation: the keyword rule); returns the
    turn record and the new state's snapshot."""
    env = VitalEnv()
    result = run_episode(env, agent, Task("edit", base_state, instruction), rounds=1)
    return result["turns"][0], snapshot(env, env.state)


def run_step(agent, episode: Episode | None, base_state: str, instruction: str) -> tuple[Episode, dict, dict]:
    """One step of the repository's step loop (synth_rl/loop.py): the agent replies to the conversation so far,
    which holds its earlier steps and the audio of each result, and the reply is applied."""
    if episode is None:
        episode = Episode(VitalEnv(), Task("edit", base_state, instruction))
    step = episode.step(agent.complete(episode.messages))
    reason = str((first_json(step["reply"]) or {}).get("reason", ""))
    if episode.stopped == "unreadable":
        reason = "The reply could not be read as an edit."
    turn = {"reason": reason, "clipped": step["clipped"], "done": episode.stopped == "done"}
    return episode, turn, snapshot(episode.env, episode.state)


def event(name: str, data: dict) -> dict:
    return {"event": name, "data": json.dumps(data)}


@app.post("/api/chat")
async def chat(body: ChatBody):
    return EventSourceResponse(chat_events(body), ping=15)


async def chat_events(body: ChatBody):
    async with chat_lock:
        turn_id = session.new_turn_id()
        with session.lock:
            from_rev = session.current.rev if body.from_rev is None else body.from_rev
        yield event("start", {"turn_id": turn_id, "agent": body.agent, "rounds": body.rounds, "mute": body.mute,
                              "from_rev": from_rev})
        with session.lock:
            first = session.revisions.get(from_rev)
        if first is None:
            yield event("error", {"message": f"no revision {from_rev}"})
            return
        if body.agent != "rule" and not body.agent.startswith(CHAT_PREFIXES):
            yield event("error", {"message": f"unknown agent {body.agent!r}; see /api/agents"})
            return
        try:
            agent = await asyncio.to_thread(make_agent, body.agent, body.mute)
        except Exception as e:  # noqa: BLE001
            yield event("error", {"message": f"cannot use agent {body.agent}: {e}"})
            return
        # The keyword rule matches anywhere in the text, so it gets the new message only.
        instruction = body.text if body.agent == "rule" else with_context(session.context_for(from_rev), body.text)
        if hasattr(agent, "complete"):
            instruction += CHAT_NOTE
        base = first
        moved: dict[str, tuple[str, str]] = {}
        episode = None  # the conversation of this turn, for agents that can hold one
        try:
            for rnd in range(1, body.rounds + 1):
                yield event("listening", {"round": rnd, "rev": base.rev})
                t0 = time.time()
                if hasattr(agent, "complete"):
                    episode, turn, snap = await asyncio.to_thread(run_step, agent, episode, base.state, instruction)
                else:
                    turn, snap = await asyncio.to_thread(run_round, agent, base.state, instruction)
                secs = round(time.time() - t0, 2)
                changes = [{"name": n, "from": base.values[n], "to": snap["values"][n],
                            "from_display": base.displays[n], "to_display": snap["displays"][n]}
                           for n in moved_controls(base.values, snap["values"], MOVED)]
                proposal = {"round": rnd, "reason": turn["reason"], "changes": changes, "secs": secs,
                            "clipped": turn.get("clipped", []), "done": turn.get("done", False)}
                if not changes:
                    yield event("proposal", proposal)
                    break
                tag = "muted " if body.mute else ""
                summary = f"{tag}round {rnd}: {body.text[:60]} ({', '.join(label_of(c['name']) for c in changes)})"
                rev = session.make_revision(snap, base.rev, "agent", summary)
                if body.from_rev is None:  # a comparison branch leaves the current sound where it is
                    session.push(rev)
                for c in changes:
                    moved[c["name"]] = (moved.get(c["name"], (c["from_display"], ""))[0], c["to_display"])
                yield event("proposal", proposal)
                yield event("revision", {"round": rnd, "rev": rev.rev, "state": session.state_json(), "delta": delta(base, rev),
                                         "same_audio": rev.wav == base.wav})
                base = rev
                if episode is not None and episode.done:
                    break
        except Exception as e:  # noqa: BLE001
            if base is not first:  # keep what was reached as context for the next turn
                session.add_turn(Turn(turn_id, body.text, from_rev, base.rev, moved))
            yield event("error", {"message": f"round failed: {e}"})
            return
        session.add_turn(Turn(turn_id, body.text, from_rev, base.rev, moved))
        yield event("done", {"turn_id": turn_id, "from_rev": from_rev, "to_rev": base.rev, "delta": delta(first, base)})


# --- web page --------------------------------------------------------------------------------------------------

@app.get("/{path:path}", include_in_schema=False)
def web(path: str):
    if path == "api" or path.startswith("api/"):
        raise HTTPException(404, "no such API route")
    if not DIST.is_dir():
        if path == "":
            return PlainTextResponse("synth-rl demo server is running; the web page is not built (pnpm build in demo/web).")
        raise HTTPException(404)
    root = DIST.resolve()
    target = (root / path).resolve()
    if root != target and root not in target.parents:
        raise HTTPException(404)
    if target.is_dir():
        target = target / "index.html"
    if target.is_file():
        cache = "no-cache" if target.name == "index.html" else "public, max-age=3600"
        return FileResponse(target, headers={"Cache-Control": cache})
    if "." not in path.rsplit("/", 1)[-1] and (root / "index.html").is_file():  # client-side route
        return FileResponse(root / "index.html", headers={"Cache-Control": "no-cache"})
    raise HTTPException(404)
