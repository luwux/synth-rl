# Earlier zero-shot runs (October 2026)

These runs were made with the patch generator and prompt format as they were up to commit `06e6d7c`. In that generator the wavetable position had no effect on the sound, the filter did not act on oscillator 2, and every reverb decay setting gave the longest decay; the prompt listed 0–1 positions instead of values in units. Scores from the current code are not comparable with them.

Mean reward on 30 procedural tasks per column, one round, 95% bootstrap intervals about ±0.15–0.25. "Muted" sends the same prompt with silent audio. Leaving the patch unchanged scores 0 everywhere; — means not run.

| Agent | Edit | Edit, real presets | Edit, blind | Repair | Match |
|---|---|---|---|---|---|
| Keyword rule (never listens) | 0.648 | 0.422 | 0.426 | 0 (no change) | 0 (no change) |
| Gemini 3.8 Flash | 0.554 | 0.415 | 0.523 | — | 0.081 |
| Gemini 3.8 Flash, muted | 0.529 | 0.350 | 0.489 | — | −0.165 |
| Qwen2.5-Omni-3B | −0.068 | −0.042 | −0.148 | −0.267 | — |
| Qwen2.5-Omni-7B | −0.028 | 0.196 | 0.029 | −0.793 | — |
| Qwen2.5-Omni-7B, muted | −0.024 | 0.206 | 0.014 | −0.768 | — |
| Qwen3-Omni-30B-A3B-Instruct | 0.253 | 0.329 | 0.088 | — | — |

"Real presets" draws its starting sounds from a local library of factory and third-party Vital presets (not included), scored over three renders. "Blind" hides the current parameter values.

- **Real and muted audio score the same within noise** in every column where both were run (Gemini and Qwen2.5-Omni-7B); the largest gap is Gemini on match (+0.25, interval −0.06 to +0.56). This is a statement about these tasks and rewards: an edit task can be done from the text, so equal scores there do not show that a model cannot hear.
- **Edit tasks can be solved from text.** The keyword rule, which never listens, matches or beats every model. With parameter values shown, the instruction says which knob to turn; the edit columns measure instruction following more than hearing.
- **Repair and match need the audio**, because the target exists only as sound. On repair, Qwen2.5-Omni-7B ends further from the target than the broken sound it started from (−0.79), and the same with silent audio (−0.77); both are worse than leaving the sound alone. Qwen2.5-Omni-3B scores higher mainly because it leaves 8 of 30 patches unchanged.
- The small Qwen Omni models are near or below zero; Qwen3-Omni-30B is the only open model clearly above zero on edits.

The descriptor reward has known blind spots. A listening check of reward-selected "best" edits found, for example, that "wider" can reward detuned, wobbling unison and "brighter" can reward added distortion. Treat edit scores as a proxy.
