# synth-rl demo: design and contract

Written 2026-11-15. This file is the contract for everyone building the demo. If code and this file disagree,
fix one of them and say so in your report.

## What the demo is

A local web app. Left: a synthesizer panel in the style of Vital showing the 37 controls the agent can edit.
Right: a chat. The user types "brighter, with a shorter tail"; a listening model hears the current sound,
sets some controls, the knobs move, the new sound plays, and the model hears that result and goes on step by step
until it says it is done (or reaches the step limit). Under the
panel: the sound (waveform, spectrogram, before/after), and a "muted" switch that reruns the same instruction with
silent audio.

Optional second display: the real Vital plugin window mirroring the same state (only if the spike passes).

## Decisions already made (do not reopen)

1. **`VitalEnv` (vita, headless) is the only source of truth** for state, audio, and measured descriptors, exactly
   as in the benchmark. A synth display only receives "show this state". The web panel and the real Vital
   window are two displays of the same server state.
2. **The agent loop is the repository's own**: `synth_rl.loop.Episode`, no agent framework. One chat message =
   one episode on `Task("edit", state, instruction)`: at most `rounds` steps of `episode.step(agent.complete(episode.messages))`,
   each streamed to the browser. The model sees its earlier steps of the turn and the audio of every result, and ends
   the turn by replying `{"done": true}`. (Since 2026-11-16; before, each round was a fresh one-shot prompt that did
   not know what the previous round had done. The keyword rule, which holds no conversation, still runs that way.)
   The server appends one sentence to the message (`CHAT_NOTE` in `app.py`): a greeting or a question gets an answer
   in the `reason` of a `{"done": true}` reply and no edit. Without it, "um hi what si this" took 4 steps and moved
   8 controls; with it, 6 such messages × 3 replies were edited 3 times instead of 13, and 6 real requests × 3 were
   edited 18 of 18 times either way (`outputs/agent_loop/chat_note.py`).
3. **Listening model**: `qwen/qwen3.8-omni-flash` through OpenRouter (verified 2026-11-15 with audio input:
   correct edits, about 14 s, about $0.0006 per call). Gemini is not available (direct API: prepaid credits
   depleted, HTTP 402; through OpenRouter: HTTP 403). `rule` is the offline fallback. A local llama-server
   endpoint is supported by the same agent class but must not be started without the user's consent (64 GB).
4. **Sounds**: procedural patches (`synth_rl.procedural.random_patch`) and the init patch only. No third-party
   presets anywhere in the demo, tests, screenshots, or recordings.
5. **Existing components, not hand-written ones**: FastAPI + uvicorn + sse-starlette (server); React + Vite +
   TypeScript + Tailwind; assistant-ui for the chat; react-knob-headless for knobs; wavesurfer.js for the waveform and
   spectrogram. Hand-write only the glue and the Vital-style skin.
6. **Look**: follow Vital's visual system (dark panels, section headers, arc knobs with a glowing value arc,
   purple/teal accents, Lato + Montserrat). Colors and proportions may be taken from Vital's GPL-3.0 source skin
   (github.com/mtytel/vital), which matches this repository's license. Do not use Vital's logo, name as a brand,
   or any image asset from the Vital app. Header text: "synth-rl" only (the small "panel styled after Vital" line
   was removed 2026-11-16 at the user's request).

## Layout of files (one owner per file)

| Path | Owner | Contents |
| --- | --- | --- |
| `demo/DESIGN.md` | lead | this file |
| `synth_rl/agents.py`, `synth_rl/cli.py` (`make_agent` only) | backend | `OpenAIChatAgent` and its agent-name syntax |
| `pyproject.toml` (`[demo]` extra only) | backend | fastapi, uvicorn, sse-starlette |
| `demo/server/` | backend | FastAPI app, session, SSE |
| `demo/tests/` | backend | acceptance test at the API level |
| `demo/web/` | frontend | Vite app; `pnpm build` writes `demo/web/dist`, which the server serves at `/` |
| `demo/mirror/` | spike | real-Vital-window mirror, only if the spike passes |
| `demo/README.md` | lead | how to run |

Nothing is committed or pushed. No API key is ever printed, logged, or written to a file. Keys are read from the
environment (`OPENROUTER_API_KEY`; the user's interactive zsh has it, so start servers with `zsh -ic '...'`).

## Server

Run: `uvicorn demo.server.app:app --host 127.0.0.1 --port 8765` from the repo root with `.venv`.
One in-memory session (single local user). A session is a stack of revisions plus a cursor.

### Types (JSON)

```jsonc
// Control: one of the 37 editable parameters
{ "name": "filter_1_cutoff", "label": "filter 1 cutoff",   // short name; the prompt description without its explanation
  "group": "filter",          // "osc1" | "osc2" | "filter" | "env1" | "env2" | "chorus" | "reverb" | "delay" | "distortion" | "eq"
  "kind": "knob",             // "knob" | "switch" (names ending in _on) | "stepped" (unison voice counts)
  "value": 0.62,              // normalized 0-1, exact
  "display": "1240 Hz" }      // the value with its unit, as the model reads it ("0.48 s", "on", "3 voices"); a shape for wavetable positions ("saw to square, 18%")

// Descriptors: synth_rl.features.describe(audio)
{ "centroid_oct": 8.82, "hf_db": -17.6, "loudness_db": -15.3, "attack_s": 0.46,
  "sustain": 0.82, "tail_s": 1.47, "width": 0.14 }

// Revision: one state of the synth
{ "rev": 3, "parent": 2,
  "source": "agent",          // "init" | "new_patch" | "manual" | "agent"
  "controls": [Control, ...], // always all available controls, fixed order of EDITABLE
  "descriptors": Descriptors,
  "audio_url": "/api/wav/3.wav?v=5f2a9c01",   // opaque; changes whenever the audio changes. 44.1 kHz stereo WAV, 3 s, note C4 held 1.5 s
  "preset_name": "procedural-7a1d5006",
  "wavetables": {             // optional (absent on older servers); per oscillator, its table's keyframes for the wave view
    "osc_1": { "positions": [0, 0.25, 0.5, 0.75, 1],   // on the wave frame knob's 0-1 scale, ascending
               "frames": [[0.0, -0.04, ...], ...] },    // one 128-sample cycle per keyframe, -1..1; the wave between two keyframes is their straight blend
    "osc_2": { ... } } }

// Change: one parameter moved by an edit
{ "name": "filter_1_cutoff", "from": 0.62, "to": 0.96, "from_display": "1.2 kHz", "to_display": "9.8 kHz" }

// State: what the page needs to draw itself
{ "current": Revision, "can_undo": true, "can_redo": false,
  "history": [ { "rev": 0, "source": "new_patch", "summary": "new patch" }, ... ] }
```

### Endpoints

| Method and path | Body | Returns |
| --- | --- | --- |
| `GET /api/state` | | `State` |
| `GET /api/agents` | | `[{ "id": "openrouter:qwen/qwen3.8-omni-flash", "label": "Qwen3.8-Omni-Flash (OpenRouter)", "listens": true, "available": true, "why": "" }, { "id": "rule", "label": "Keyword rule (never listens)", "listens": false, "available": true, "why": "" }, ...]` |
| `POST /api/new_patch` | `{ "seed": 7 }` or `{ "init": true }` or `{}` (random seed) | `State` (history reset) |
| `POST /api/set` | `{ "edits": { "filter_1_cutoff": 0.8 } }` | `State`; a new revision with `source: "manual"`. Consecutive manual sets on the same single parameter within 2 s replace the previous manual revision instead of stacking |
| `POST /api/undo`, `POST /api/redo` | | `State` |
| `GET /api/wav/{rev}.wav` | | WAV, `Cache-Control: no-store`. Always use `audio_url` from a revision, never build this path |
| `GET /api/preset/{rev}.vital` | | the preset JSON as a download |
| `POST /api/open_in_vital` | `{ "rev": 3 }` | `{ "ok": true }`; writes the preset to a temp file and runs `open -a Vital <file>` |
| `POST /api/chat` | `{ "text": "...", "agent": "<id>", "rounds": 4, "mute": false }` (`rounds` = the most steps the agent may take, 1–8) | `text/event-stream`, see below |

`POST /api/chat` events (SSE `event:` name and JSON `data:`), in order:

```
event: start      data: { "turn_id": "t5", "agent": "<id>", "rounds": 2, "mute": false, "from_rev": 3 }
event: listening  data: { "round": 1, "rev": 3 }                     // the model is being called with the audio of rev 3
event: proposal   data: { "round": 1, "reason": "...", "changes": [Change, ...], "secs": 13.9, "clipped": [], "done": false }   // clipped = names whose requested value lay outside the range; done = the model said it is done
event: revision   data: { "round": 1, "rev": 5, "state": State, "delta": { "centroid_oct": 0.51, "tail_s": -0.22, ... }, "same_audio": false }   // delta = descriptors(after) - descriptors(before this round); rev = the revision this step made (in a comparison branch it is not state.current); same_audio = the render is byte for byte the audio the round started from
... (same three for each further step; a step whose proposal has no changes, usually the model's "done", emits proposal with changes: [] and no revision, and ends the turn)
event: done       data: { "turn_id": "t5", "from_rev": 3, "to_rev": 5, "delta": { ...total... } }
event: error      data: { "message": "..." }                         // then the stream ends; state is whatever was reached
```

The filter envelope is two cutoffs, not a depth: `filter_1_cutoff` (envelope 2 at zero) and
`filter_1_cutoff_at_env_2_peak` (envelope 2 at its peak), both 0-1 on the cutoff's scale. Setting either leaves the
other where it is; while they are equal (no movement) the peak follows the cutoff, and then it is not listed as a
change of its own. The filter display draws the swept range.

Follow-ups: `propose` keeps no history. The server puts the conversation in front of the instruction text: one
numbered line per earlier turn on the way to the current revision (`The user asked: "<text>". You changed: <name>
<from_display> -> <to_display>, ...`) and per manual change (`The user changed by hand: ...`), the last 12 at most,
then `Earlier requests still apply unless the new one contradicts them.` and `Now: <new text>`. (Until 2026-11-16
only the last turn was included, and a request two turns back was undone: see `outputs/wavetable_fix/context_replay.py`.)
Undo is the revision stack, not a model call.

`mute: true` builds the agent with `mute_audio=True` (same prompt, silent audio). The page uses this for the
"compare with muted" action: it branches from the same `from_rev` (the server accepts `"from_rev": N` in the chat
body, default = current) and shows both results side by side. The branch result is a revision that can be played and downloaded, but it does not become the current sound: the panel, the player, and undo stay on the turn's own result.

## Web page

Desktop layout, min width 1280. Left 2/3 synth, right 1/3 chat. Dark only.

- **Header**: "synth-rl", patch name, buttons: New patch, Init patch, Undo, Redo, Download .vital, Open in Vital.
- **Synth panel**: sections in Vital's order: OSC 1, OSC 2, FILTER 1, ENV 1 (amp), ENV 2 (filter), then effects
  row: CHORUS, DELAY, DISTORTION, EQ, REVERB. Each section has its on/off switch in the header where one exists.
  Every one of the 37 controls is visible without scrolling at 1440x900. Knobs are draggable (vertical drag,
  shift = fine); releasing sends `/api/set`. Controls changed by the last agent round glow and animate from old to
  new value (about 600 ms); a small ghost tick marks the previous value until the next change.
  ENV sections also draw the ADSR curve from the values, FILTER 1 an approximate response. OSC 1 and OSC 2 show,
  left of their knobs, the single cycle the oscillator plays at its FRAME position (blended from
  `Revision.wavetables`); the view reads the FRAME knob's own displayed value, so it morphs during a drag and the
  glide after an agent round, and it is drawn dimmed while OSC 2 is off.
- **Sound strip** under the panel: waveform + spectrogram of the current revision (wavesurfer), Play (space),
  A/B toggle between "before this turn" and "now", autoplay after each agent round (toggle, default on).
- **No descriptor readout** (removed 2026-11-16: the seven descriptors miss too much to be shown as "what
  happened"; the API still sends them). A round card shows two facts only: a notice when `same_audio` is true,
  and a warning when the round's level dropped by 12 dB or more.
- **Chat** (assistant-ui): user bubble; assistant turn made of per-round cards: "Listening…" spinner, then the
  reason sentence, then a compact list of changes (`filter 1 cutoff  1.2 kHz → 9.8 kHz`), each row clickable
  to flash the knob. Composer at the bottom with: agent selector (from `/api/agents`), max steps (1, 2, 4, 6; default 4),
  "muted" switch, and four example chips: "brighter, with a shorter tail", "wider and more spacious",
  "more plucky", "darker and dirtier". After a turn: a "Compare with muted audio" button on that turn.
- Empty, loading, and error states must be designed, not default.

## Acceptance (the demo is "right" when all pass)

1. API level (`demo/tests/acceptance.py`, no browser): new patch with seed 3; chat "brighter, with a shorter
   tail", at most 2 steps; with the OpenRouter agent: `centroid_oct` up and `tail_s` not up, at least one change, final
   revision audio differs from the first; with `rule`: stream completes and state is consistent. Undo returns the
   exact earlier controls. Manual set changes the audio.
2. Browser level (chrome-devtools CLI on Dia, never another browser tool): load the page, click an example chip,
   wait for the turn to finish, screenshot. Knob values on screen equal `/api/state`. Play works. Undo works.
3. No console errors, no failed requests, no key in any log or response.

## Status of the unknowns

- Real Vital window (spike, 2026-11-15): works as a one-way mirror, `demo/mirror/vital_mirror.py`. The whole
  preset is pushed as the hosted VST3's state through pedalboard in 15-20 ms and the open window redraws.
  Verified by window captures on the VOICE tab only; the EFFECTS tab (14 of the 37 controls) still needs a manual
  look. The plugin's own sound matches vita except for chorus and reverb phase; the mirror plays nothing.
- vita stamps states with synth_version 99999.9.9, which Vital refuses. `synth_rl.vital_env.preset_file` stamps
  1.5.5 for every exported file (CLI `final.vital`, `/api/preset`, "Open in Vital").
- `open -a Vital <file>` is ignored by Vital. "Open in Vital" runs `open -n -a Vital --args <file>`, which starts
  a new Vital instance with the preset.
- vita can write `reverb_decay_time` above Vital's maximum of 6; real Vital clamps it. Not fixed.
- Agent variants: `openrouter:qwen/qwen3.8-omni-flash` (reasoning off, about 2 s per step) and the same id with
  `+think` (reasoning on, about 30 s). Default is decided by `outputs/qwen38/` (30 tasks x 4 task sets).
