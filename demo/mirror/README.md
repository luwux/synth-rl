# Real Vital window mirror (optional, one-way)

Opens the real Vital plugin window and makes it follow the demo server. Run from the repo root while the server runs:

    uv run --no-project --python 3.12 --with pedalboard python demo/mirror/vital_mirror.py [--server http://127.0.0.1:8765]

It needs `/Library/Audio/Plug-Ins/VST3/Vital.vst3` (tested with Vital 1.5.5). It polls `/api/state` every 0.3 s. When `current.rev` or the control values change, it loads `/api/preset/{rev}.vital` as the plugin's whole state, which takes 15–20 ms, and the open window redraws. The window title is "Pedalboard". To quit, close the window, press Ctrl-C, or send SIGTERM.

Limits: knobs turned in this window are not sent back, and the next server change overwrites them. Nothing is played. Vital's sound matches vita's within chorus and reverb phase variation. States saved before 2026-11-16 can carry a reverb decay time above Vital's maximum of 6 (a mapping error since fixed in `synth_rl/vital_env.py`). Vital accepts the value but clamps it to 6 once audio runs, and the knob then reads maximum. The filter graph shows live modulation, so it can differ from the cutoff value until a note plays.
