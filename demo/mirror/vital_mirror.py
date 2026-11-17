"""One-way mirror: the real Vital plugin window follows the demo server's current state.

Hosts the Vital VST3 with pedalboard and opens its editor. A background thread polls `GET /api/state`; when the
current revision (or its control values) changes, it fetches `GET /api/preset/{rev}.vital` and loads it into the
plugin as its whole state, which redraws the open editor. Nothing flows back to the server: knob moves made in this
window are not sent anywhere and are overwritten by the next change on the server.

Run from the repository root (dependencies are not in the repo's .venv):

    uv run --no-project --python 3.12 --with pedalboard python demo/mirror/vital_mirror.py

Closing the window, Ctrl-C, or SIGTERM ends the process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import struct
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pedalboard

DEFAULT_PLUGIN = "/Library/Audio/Plug-Ins/VST3/Vital.vst3"

# --- Vital preset JSON -> pedalboard `raw_state` of the Vital VST3 ----------------------------------------------
#
# `raw_state` of a JUCE VST3 plugin is "VC2!" + uint32 size + XML <VST3PluginState><IComponent>...</IComponent>.
# IComponent holds JUCE's own base64 of: VstW header (16 bytes) + VST2 fxBank header (160 bytes, big-endian sizes
# at offsets 4 and 156) + Vital's getStateInformation (the preset JSON, a NUL, then JUCE's private-data trailer).

_B64 = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
_B64_ARR = np.frombuffer(_B64.encode(), np.uint8)
_B64_LUT = np.full(256, 255, np.uint8)
_B64_LUT[_B64_ARR] = np.arange(64, dtype=np.uint8)


def _b64_decode(s: str) -> bytes:
    size, data = s.split(".", 1)
    v = _B64_LUT[np.frombuffer(data.encode(), np.uint8)]
    bits = ((v[:, None] >> np.arange(6)) & 1).astype(np.uint8).ravel()
    return np.packbits(bits, bitorder="little").tobytes()[: int(size)]


def _b64_encode(b: bytes) -> str:
    bits = np.unpackbits(np.frombuffer(b, np.uint8), bitorder="little")
    n = (len(bits) + 5) // 6
    bits = np.concatenate([bits, np.zeros(n * 6 - len(bits), np.uint8)])
    v = (bits.reshape(n, 6).astype(np.int64) << np.arange(6)).sum(1)
    return f"{len(b)}." + _B64_ARR[v].tobytes().decode()


def _split(raw: bytes) -> tuple[bytes, bytes, bytes, bytes]:
    """-> (VstW header, fxBank header, plugin JSON, trailer after the JSON's NUL)."""
    if raw[:4] != b"VC2!":
        raise ValueError("not a JUCE VST3 state")
    xml = raw[8:].rstrip(b"\0").decode()
    m = re.search(r"<IComponent>([^<]*)</IComponent>", xml)
    if not m:
        raise ValueError("no IComponent in state")
    comp = _b64_decode(m.group(1).strip())
    if comp[:4] != b"VstW" or comp[16:20] != b"CcnK":
        raise ValueError("unexpected component layout")
    mem = comp[176:]
    end = mem.index(b"\0")
    return comp[:16], comp[16:176], mem[:end], mem[end + 1 :]


def preset_to_raw_state(template: bytes, preset: str) -> bytes:
    """Wrap a Vital preset JSON string in the container of `template` (a raw_state taken from the same plugin)."""
    head, bank, own_json, trailer = _split(template)
    own = json.loads(own_json)
    data = json.loads(preset)
    # vita writes synth_version 99999.9.9; Vital refuses presets "created with a newer version". Stamp the
    # installed plugin's own version instead.
    data["synth_version"] = own.get("synth_version", "1.5.5")
    data.setdefault("tuning", own.get("tuning"))
    mem = json.dumps(data, separators=(",", ":")).encode() + b"\0" + trailer
    bank = bytearray(bank)
    struct.pack_into(">i", bank, 4, 160 - 8 + len(mem))
    struct.pack_into(">i", bank, 156, len(mem))
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?> <VST3PluginState><IComponent>'
        + _b64_encode(head + bytes(bank) + mem)
        + "</IComponent></VST3PluginState>"
    ).encode() + b"\0"
    return b"VC2!" + struct.pack("<I", len(xml) - 1) + xml


def preset_name_of(raw: bytes) -> str:
    return json.loads(_split(raw)[2]).get("preset_name", "")


# --- polling ----------------------------------------------------------------------------------------------------


def _get(url: str, timeout: float = 5.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def follow(plugin, template: bytes, server: str, interval: float, stop: threading.Event) -> None:
    last_key = None
    down = False
    while not stop.is_set():
        try:
            state = json.loads(_get(f"{server}/api/state"))
            cur = state["current"]
            # Key on the control values too: a revision may be replaced in place (manual sets within 2 s).
            values = json.dumps([[c["name"], c["value"]] for c in cur.get("controls", [])])
            key = (cur["rev"], hashlib.sha1(values.encode()).hexdigest())
            if down:
                print("server is back", flush=True)
                down = False
            if key != last_key:
                preset = _get(f"{server}/api/preset/{cur['rev']}.vital").decode()
                t = time.time()
                plugin.raw_state = preset_to_raw_state(template, preset)
                last_key = key
                name = cur.get("preset_name") or json.loads(preset).get("preset_name", "")
                print(f"rev {cur['rev']} ({name}) shown in {1000 * (time.time() - t):.0f} ms", flush=True)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if not down:
                print(f"waiting for the server at {server} ({e})", flush=True)
                down = True
        except Exception as e:  # a bad response must not kill the window
            print(f"skipped an update: {type(e).__name__}: {e}", flush=True)
        stop.wait(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--interval", type=float, default=0.3, help="seconds between polls")
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN)
    args = ap.parse_args()

    plugin = pedalboard.load_plugin(args.plugin)
    template = plugin.raw_state
    _split(template)  # fail early if the plugin's state layout is not the one this file knows

    stop = threading.Event()  # set: stop polling and close the window
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    threading.Thread(target=follow, args=(plugin, template, args.server.rstrip("/"), args.interval, stop),
                     daemon=True).start()
    print(f"Vital mirror following {args.server}; close the window or press Ctrl-C to quit", flush=True)
    plugin.show_editor(stop)  # main thread; returns when the window is closed or `stop` is set
    stop.set()


if __name__ == "__main__":
    sys.exit(main())
