"""API-level acceptance for the demo server (demo/DESIGN.md, "Acceptance" 1, plus the key check of 3).

    .venv/bin/python demo/tests/acceptance.py --agent rule
    zsh -ic 'cd <repo> && .venv/bin/python demo/tests/acceptance.py --agent openrouter:qwen/qwen3.8-omni-flash'

Without --agent: `rule`, plus the OpenRouter agent when OPENROUTER_API_KEY is set. The script starts its own
server on a free port and stops it at the end. Prints one PASS/FAIL line per check; exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from synth_rl import vital_env  # noqa: E402
from synth_rl.vital_env import EDITABLE  # noqa: E402

OPENROUTER = "openrouter:qwen/qwen3.8-omni-flash"
INSTRUCTION = "brighter, with a shorter tail"


class Client:
    def __init__(self, base: str):
        self.base = base
        self.bodies: list[bytes] = []  # every response body, for the key check

    def request(self, method: str, path: str, body: dict | None = None, timeout: float = 30):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw, status, headers = resp.read(), resp.status, dict(resp.headers)
        except urllib.error.HTTPError as e:
            raw, status, headers = e.read(), e.code, dict(e.headers)
        self.bodies.append(raw)
        return status, raw, headers

    def json(self, method: str, path: str, body: dict | None = None):
        status, raw, _ = self.request(method, path, body)
        if status != 200:
            raise AssertionError(f"{method} {path} -> HTTP {status}: {raw[:200]!r}")
        return json.loads(raw)

    def sse(self, path: str, body: dict, timeout: float = 300) -> list[tuple[str, dict]]:
        """Events of a POST text/event-stream, parsed line by line (sse-starlette uses CRLF and ping comments)."""
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        events, name, data, raw = [], "message", [], []
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            assert resp.headers.get("content-type", "").startswith("text/event-stream"), resp.headers
            for line in resp:
                raw.append(line)
                line = line.decode().rstrip("\r\n")
                if line.startswith(":"):
                    continue
                if line == "":
                    if data:
                        events.append((name, json.loads("\n".join(data))))
                    name, data = "message", []
                elif line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())
        if data:
            events.append((name, json.loads("\n".join(data))))
        self.bodies.append(b"".join(raw))
        return events


class Checks:
    def __init__(self):
        self.failed = 0

    def __call__(self, ok: bool, name: str, detail: str = "") -> bool:
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}", flush=True)
        self.failed += not ok
        return ok


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def controls_map(state: dict) -> dict[str, float]:
    return {c["name"]: c["value"] for c in state["current"]["controls"]}


def wav_info(raw: bytes):
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return sr, audio.shape[1], audio.shape[0] / sr


def run_agent(c: Client, check: Checks, agent: str) -> None:
    listens = agent != "rule"
    print(f"\n== agent {agent}", flush=True)
    s0 = c.json("POST", "/api/new_patch", {"seed": 3})
    cur = s0["current"]
    names = [x["name"] for x in cur["controls"]]
    check(names == list(EDITABLE), f"new patch seed 3: {len(EDITABLE)} controls in EDITABLE order", f"{len(names)} controls")
    check(cur["source"] == "new_patch" and len(s0["history"]) == 1 and not s0["can_undo"] and not s0["can_redo"],
          "new patch: history reset", json.dumps(s0["history"]))
    check(all(0.0 <= x["value"] <= 1.0 for x in cur["controls"])
          and {x["kind"] for x in cur["controls"]} == {"knob", "switch", "stepped"}, "control values and kinds")
    status, first_wav, headers = c.request("GET", cur["audio_url"])
    sr, ch, dur = wav_info(first_wav)
    check(status == 200 and sr == 44100 and ch == 2 and abs(dur - 3.0) < 0.01, "audio: 44.1 kHz stereo 3 s WAV",
          f"{sr} Hz, {ch} ch, {dur:.2f} s")
    cache = {k.lower(): v for k, v in dict(headers).items()}.get("cache-control", "")
    check("no-store" in cache, "audio is never cached by the browser", cache)
    first_controls = controls_map(s0)

    t0 = time.time()
    events = c.sse("/api/chat", {"text": INSTRUCTION, "agent": agent, "rounds": 2, "mute": False})
    secs = time.time() - t0
    kinds = [e for e, _ in events]
    print(f"      events: {' '.join(kinds)}  ({secs:.1f} s)")
    for e, d in events:
        if e == "proposal":
            print(f"      round {d['round']} ({d['secs']} s): {d['reason'][:120]}")
            for ch_ in d["changes"]:
                print(f"        {ch_['name']:22s} {ch_['from_display'] or round(ch_['from'], 3)} -> "
                      f"{ch_['to_display'] or round(ch_['to'], 3)}")
        if e == "error":
            print(f"      error: {d['message'][:300]}")
    check("error" not in kinds and kinds[:1] == ["start"] and kinds[-1:] == ["done"], "chat stream completes")

    # Grammar: start (listening proposal revision?)+ done; a proposal without a revision ends the turn.
    ok, i, rounds, prev_rev = True, 1, [], events[0][1]["from_rev"] if events else None
    by_rev = {cur["rev"]: s0["current"]}
    while ok and i < len(events) and events[i][0] == "listening":
        lst = events[i][1]
        ok &= lst["rev"] == prev_rev and lst["round"] == len(rounds) + 1
        if i + 1 >= len(events) or events[i + 1][0] != "proposal":
            ok = False
            break
        prop = events[i + 1][1]
        if i + 2 < len(events) and events[i + 2][0] == "revision":
            revn = events[i + 2][1]
            after = revn["state"]["current"]
            before = by_rev[prev_rev]
            bmap = {x["name"]: x for x in before["controls"]}
            amap = {x["name"]: x for x in after["controls"]}
            # A peak that only followed the cutoff (equal to it before and after) is not listed as a change.
            moved = set(vital_env.moved({n: x["value"] for n, x in bmap.items()}, {n: x["value"] for n, x in amap.items()}, 1e-6))
            ok &= bool(prop["changes"]) and moved == {x["name"] for x in prop["changes"]}
            ok &= all(x["from"] == bmap[x["name"]]["value"] and x["to"] == amap[x["name"]]["value"]
                      and x["from_display"] == bmap[x["name"]]["display"] and x["to_display"] == amap[x["name"]]["display"]
                      for x in prop["changes"])
            ok &= after["parent"] == prev_rev and after["source"] == "agent"
            ok &= all(abs(revn["delta"][k] - (after["descriptors"][k] - before["descriptors"][k])) < 1e-9
                      for k in after["descriptors"])
            same = c.request("GET", after["audio_url"])[1] == c.request("GET", before["audio_url"])[1]
            ok &= revn["same_audio"] == same
            by_rev[after["rev"]] = after
            rounds.append(after["rev"])
            prev_rev = after["rev"]
            i += 3
        else:
            ok &= prop["changes"] == []
            i += 2
            break
    done = events[-1][1] if kinds[-1:] == ["done"] else {}
    ok &= i == len(events) - 1 and done.get("to_rev") == prev_rev and done.get("from_rev") == cur["rev"]
    ok &= 1 <= len([k for k in kinds if k == "listening"]) <= 2
    check(ok, "event order and payloads (changes = exact control moves, deltas = descriptor differences, same_audio = served bytes equal)",
          f"revisions {rounds}")
    proposals = [d for e, d in events if e == "proposal"]
    check(all(isinstance(d.get("done"), bool) for d in proposals) and (listens or not any(d["done"] for d in proposals)),
          "every proposal says whether the agent declared itself done", str([d.get("done") for d in proposals]))
    final = by_rev[prev_rev]
    check(bool(done) and all(abs(done["delta"][k] - (final["descriptors"][k] - cur["descriptors"][k])) < 1e-9
                             for k in cur["descriptors"]), "done.delta = final - first descriptors")

    st = c.json("GET", "/api/state")
    check(st["current"]["rev"] == done.get("to_rev") and st["current"]["controls"] == final["controls"]
          and st["current"]["descriptors"] == final["descriptors"], "state after the turn is the last revision")
    status, final_wav, _ = c.request("GET", final["audio_url"])
    if listens:
        dc = final["descriptors"]["centroid_oct"] - cur["descriptors"]["centroid_oct"]
        dt = final["descriptors"]["tail_s"] - cur["descriptors"]["tail_s"]
        n_changes = sum(len(d["changes"]) for e, d in events if e == "proposal")
        check(dc > 0, "centroid_oct up", f"{dc:+.3f} oct")
        check(dt <= 0, "tail_s not up", f"{dt:+.3f} s")
        check(n_changes >= 1, "at least one change", str(n_changes))
        check(status == 200 and final_wav != first_wav, "final audio differs from the first")
    else:
        check(status == 200 and (final_wav != first_wav) == bool(rounds), "final audio consistent with the revisions")

    # Undo back to the start, exact controls; redo forward again.
    for _ in rounds:
        st = c.json("POST", "/api/undo")
    check(st["current"]["rev"] == cur["rev"] and controls_map(st) == first_controls and not st["can_undo"],
          "undo returns the exact earlier controls", f"rev {st['current']['rev']}")
    for _ in rounds:
        st = c.json("POST", "/api/redo")
    check(st["current"]["rev"] == prev_rev and st["current"]["controls"] == final["controls"] and not st["can_redo"],
          "redo returns to the turn's result")

    # Muted rerun from the same starting revision: a branch that leaves the current sound where it is.
    ev = c.sse("/api/chat", {"text": INSTRUCTION, "agent": agent, "rounds": 1, "mute": True,
                             "from_rev": cur["rev"]})
    k = [e for e, _ in ev]
    revs = [d for e, d in ev if e == "revision"]
    played = [c.request("GET", f"/api/wav/{d['rev']}.wav")[0] for d in revs]
    reason = next((d["reason"] for e, d in ev if e == "proposal"), "")
    check(k[:1] == ["start"] and k[-1:] == ["done"] and ev[0][1]["mute"] is True and ev[0][1]["from_rev"] == cur["rev"]
          and all(d["state"]["current"]["rev"] == prev_rev and d["rev"] != prev_rev for d in revs)
          and all(status == 200 for status in played) and c.json("GET", "/api/state")["current"]["rev"] == prev_rev,
          "muted branch from from_rev leaves the current sound alone and can be played",
          f"{' '.join(k)}; {len(revs)} revision(s); reason: {reason[:100]}")

    # Manual set: new revision, audio changes; a second set of the same parameter within 2 s replaces it.
    st = c.json("GET", "/api/state")
    base = st["current"]
    old = controls_map(st)["filter_1_cutoff"]
    st1 = c.json("POST", "/api/set", {"edits": {"filter_1_cutoff": 0.15 if old > 0.5 else 0.95}})
    m1 = st1["current"]
    _, wav_a, _ = c.request("GET", base["audio_url"])
    _, wav_b, _ = c.request("GET", m1["audio_url"])
    check(m1["source"] == "manual" and m1["parent"] == base["rev"] and wav_a != wav_b
          and len(st1["history"]) == len(st["history"]) + 1, "manual set makes a new revision and changes the audio",
          f"cutoff {old:.3f} -> {controls_map(st1)['filter_1_cutoff']:.3f}")
    st2 = c.json("POST", "/api/set", {"edits": {"filter_1_cutoff": 0.5}})
    check(st2["current"]["rev"] != m1["rev"] or st2["current"]["audio_url"] != m1["audio_url"],
          "a revision replaced in place gets a new audio URL", st2["current"]["audio_url"])
    m2 = st2["current"]
    check(len(st2["history"]) == len(st1["history"]) and m2["rev"] != m1["rev"] and m2["parent"] == base["rev"]
          and st2["history"][-1]["rev"] == m2["rev"], "same-parameter set within 2 s replaces the manual revision")
    st3 = c.json("POST", "/api/set", {"edits": {"reverb_dry_wet": 0.4}})
    check(len(st3["history"]) == len(st2["history"]) + 1, "set of another parameter stacks")
    st4 = c.json("POST", "/api/undo")
    check(st4["current"]["controls"] == m2["controls"], "undo after manual sets")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", action="append", help="agent id (repeatable)")
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    agents = args.agent or (["rule", OPENROUTER] if key else ["rule"])
    port = free_port()
    log = tempfile.NamedTemporaryFile(prefix="synth-rl-acceptance-", suffix=".log", delete=False)
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "demo.server.app:app", "--host", "127.0.0.1",
                               "--port", str(port)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    check = Checks()
    c = Client(f"http://127.0.0.1:{port}")
    try:
        for _ in range(300):
            try:
                c.json("GET", "/api/state")
                break
            except Exception:  # noqa: BLE001
                if server.poll() is not None:
                    raise SystemExit(f"server exited; log: {log.name}")
                time.sleep(0.1)
        print(f"server on port {port}, log {log.name}")
        listed = c.json("GET", "/api/agents")
        ids = {a["id"]: a for a in listed}
        check(all(set(a) == {"id", "label", "listens", "available", "why"} for a in listed)
              and ids.get("rule", {}).get("available") is True, "/api/agents shape, rule available",
              ", ".join(f"{a['id']}={'yes' if a['available'] else 'no: ' + a['why']}" for a in listed))
        status, raw, headers = c.request("GET", "/")
        check(status == 200, "GET / answers", f"{headers.get('content-type')}: {raw[:80]!r}")
        check(c.request("GET", "/api/nope")[0] == 404 and c.request("GET", "/api/wav/999999.wav")[0] == 404,
              "unknown API path and revision give 404")
        st = c.json("GET", "/api/state")
        status, raw, headers = c.request("GET", f"/api/preset/{st['current']['rev']}.vital")
        check(status == 200 and "attachment" in headers.get("content-disposition", "")
              and "settings" in json.loads(raw), "preset download", headers.get("content-disposition", ""))
        for agent in agents:
            if agent in ids and not ids[agent]["available"]:
                check(False, f"agent {agent} available", ids[agent]["why"])
                continue
            try:
                run_agent(c, check, agent)
            except Exception as e:  # noqa: BLE001
                check(False, f"agent {agent}: unexpected exception", repr(e)[:300])
    finally:
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    server_log = Path(log.name).read_bytes()
    if key:
        leaked = any(key.encode() in b for b in c.bodies) or key.encode() in server_log
        check(not leaked, "API key absent from every response and the server log")
    errors = [ln for ln in server_log.decode("utf-8", "replace").splitlines() if "Traceback" in ln or " 500 " in ln]
    check(not errors, "no tracebacks or HTTP 500 in the server log", "; ".join(errors[:3]))
    print(f"\n{'ALL PASS' if not check.failed else f'{check.failed} FAILED'}")
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
