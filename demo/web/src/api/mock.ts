// Development fixtures that follow demo/DESIGN.md, used only when the page is opened with ?mock=1.
// Not a model of Vital: displays and audio are rough stand-ins so the page can be built without the server.

import type {
  AgentInfo,
  Api,
  Change,
  ChatEvent,
  ChatRequest,
  Control,
  ControlKind,
  Delta,
  Descriptors,
  Group,
  Revision,
  RevisionSource,
  State,
  Wavetable,
} from "./types";
import { DESCRIPTOR_KEYS } from "./types";

const EDITABLE: [string, string][] = [
  ["osc_1_level", "oscillator 1 level"],
  ["osc_1_wave_frame", "oscillator 1 wavetable position"],
  ["osc_1_unison_voices", "oscillator 1 unison voice count"],
  ["osc_1_unison_detune", "oscillator 1 unison detune amount"],
  ["osc_1_stereo_spread", "oscillator 1 unison stereo spread"],
  ["osc_2_on", "oscillator 2 switch"],
  ["osc_2_level", "oscillator 2 level"],
  ["osc_2_wave_frame", "oscillator 2 wavetable position"],
  ["osc_2_unison_voices", "oscillator 2 unison voice count"],
  ["osc_2_unison_detune", "oscillator 2 unison detune amount"],
  ["filter_1_on", "filter 1 switch"],
  ["filter_1_cutoff", "filter 1 cutoff while envelope 2 is at zero"],
  ["filter_1_cutoff_at_env_2_peak", "filter 1 cutoff when envelope 2 is at its peak"],
  ["filter_1_resonance", "filter 1 resonance"],
  ["filter_1_drive", "filter 1 drive"],
  ["filter_1_mix", "filter 1 dry/wet"],
  ["env_1_attack", "amp envelope attack"],
  ["env_1_decay", "amp envelope decay"],
  ["env_1_sustain", "amp envelope sustain"],
  ["env_1_release", "amp envelope release"],
  ["env_2_attack", "envelope 2 attack time (envelope 2 moves the filter cutoff)"],
  ["env_2_decay", "envelope 2 decay"],
  ["env_2_sustain", "envelope 2 sustain"],
  ["chorus_on", "chorus switch"],
  ["chorus_dry_wet", "chorus mix"],
  ["reverb_on", "reverb switch"],
  ["reverb_dry_wet", "reverb mix"],
  ["reverb_decay_time", "reverb decay time (0 short, 1 long)"],
  ["reverb_size", "reverb size"],
  ["delay_on", "delay switch"],
  ["delay_dry_wet", "delay mix"],
  ["distortion_on", "distortion switch"],
  ["distortion_drive", "distortion drive"],
  ["distortion_mix", "distortion mix"],
  ["eq_on", "EQ switch"],
  ["eq_low_gain", "EQ low shelf gain"],
  ["eq_high_gain", "EQ high shelf gain"],
];

function groupOf(name: string): Group {
  if (name.startsWith("osc_1")) return "osc1";
  if (name.startsWith("osc_2")) return "osc2";
  if (name.startsWith("filter")) return "filter";
  if (name.startsWith("env_1")) return "env1";
  if (name.startsWith("env_2")) return "env2";
  return name.split("_")[0] as Group;
}
function kindOf(name: string): ControlKind {
  if (name.endsWith("_on")) return "switch";
  if (name.endsWith("_unison_voices")) return "stepped";
  return "knob";
}

// Display text in the server's style (synth_rl/units.py): three significant digits and a unit.
function num(x: number): string {
  if (Math.abs(x) < 5e-5) return "0";
  const d = Math.min(4, 2 - Math.floor(Math.log10(Math.abs(x))));
  if (d <= 0) return String(Math.round(x / 10 ** -d) * 10 ** -d);
  return x.toFixed(d).replace(/0+$/, "").replace(/\.$/, "");
}
const cutoffHz = (v: number) => 440 * 2 ** ((8 + v * 128 - 69) / 12);
function display(name: string, v: number): string {
  if (name.endsWith("_on")) return v > 0.5 ? "on" : "off";
  if (name.endsWith("_unison_voices")) {
    const n = Math.round(v * 15) + 1;
    return `${n} voice${n === 1 ? "" : "s"}`;
  }
  if (name.includes("cutoff")) return `${num(cutoffHz(v))} Hz`;
  if (/^env_\d_(attack|decay|release)$/.test(name)) return `${num(v ** 4)} s`;
  if (name === "reverb_decay_time") return `${num(2 ** v)} s`;
  if (name.endsWith("_wave_frame")) {
    // as synth_rl.wavetable.position_text
    const shapes = ["sine", "triangle", "saw", "square", "pulse"];
    const x = Math.min(1, Math.max(0, v)) * 4;
    const i = Math.min(Math.floor(x), 3);
    const mix = Math.round((x - i) * 100);
    return mix === 0 || mix === 100 ? shapes[i + mix / 100] : `${shapes[i]} to ${shapes[i + 1]}, ${mix}%`;
  }
  if (name.startsWith("eq_")) return `${num((v - 0.5) * 30)} dB`;
  if (name.endsWith("_drive")) return `${num(v * 20)} dB`;
  return `${num(v * 100)} %`;
}

const INIT: Record<string, number> = {
  osc_1_level: 0.707,
  osc_1_wave_frame: 0,
  osc_1_unison_voices: 0,
  osc_1_unison_detune: 0.2,
  osc_1_stereo_spread: 1,
  osc_2_on: 0,
  osc_2_level: 0.707,
  osc_2_wave_frame: 0,
  osc_2_unison_voices: 0,
  osc_2_unison_detune: 0.2,
  filter_1_on: 0,
  filter_1_cutoff: 0.5,
  filter_1_cutoff_at_env_2_peak: 0.5,
  filter_1_resonance: 0.5,
  filter_1_drive: 0,
  filter_1_mix: 1,
  env_1_attack: 0.149,
  env_1_decay: 0.6,
  env_1_sustain: 1,
  env_1_release: 0.548,
  env_2_attack: 0.149,
  env_2_decay: 0.6,
  env_2_sustain: 1,
  chorus_on: 0,
  chorus_dry_wet: 0.5,
  reverb_on: 0,
  reverb_dry_wet: 0.25,
  reverb_decay_time: 0.5,
  reverb_size: 0.5,
  delay_on: 0,
  delay_dry_wet: 0.3,
  distortion_on: 0,
  distortion_drive: 0.25,
  distortion_mix: 1,
  eq_on: 0,
  eq_low_gain: 0.5,
  eq_high_gain: 0.5,
};

function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) % 100000) / 100000;
  };
}

function randomPatch(seed: number): Record<string, number> {
  const r = rng(seed * 2654435761);
  const v: Record<string, number> = {};
  for (const [name] of EDITABLE) {
    if (name.endsWith("_on")) v[name] = r() < 0.45 ? 1 : 0;
    else if (name.endsWith("_unison_voices")) v[name] = Math.round(r() * 7) / 15;
    else v[name] = Math.round(r() * 1000) / 1000;
  }
  v.filter_1_on = 1;
  if (r() < 0.5) v.filter_1_cutoff_at_env_2_peak = v.filter_1_cutoff;
  v.osc_1_level = 0.55 + 0.3 * r();
  v.env_1_attack = 0.05 + 0.25 * r();
  return v;
}

// ---- tiny renderer: saw + one-pole lowpass + ADSR + crude reverb, 3 s, note C4 held 1.5 s ----

const SR = 44100;
const DUR = 3;
const HOLD = 1.5;
const envTime = (v: number) => 32 * Math.pow(v, 4);

function render(p: Record<string, number>): Float32Array[] {
  const n = SR * DUR;
  const L = new Float32Array(n);
  const R = new Float32Array(n);
  const f0 = 261.63;
  const voices = Math.round(p.osc_1_unison_voices * 15) + 1;
  const det = p.osc_1_unison_detune * 0.02;
  const a = Math.max(envTime(p.env_1_attack), 0.002);
  const d = Math.max(envTime(p.env_1_decay), 0.005);
  const s = p.env_1_sustain;
  const rel = Math.max(envTime(p.env_1_release), 0.005);
  const a2 = Math.max(envTime(p.env_2_attack), 0.002);
  const d2 = Math.max(envTime(p.env_2_decay), 0.005);
  const s2 = p.env_2_sustain;
  const cutoffBase = p.filter_1_on > 0.5 ? 8.18 * Math.pow(2, (8 + p.filter_1_cutoff * 128) / 12) : 20000;
  const envSemis = (p.filter_1_cutoff_at_env_2_peak - p.filter_1_cutoff) * 128;
  const phases = Array.from({ length: voices }, (_, i) => i / voices);
  let lpL = 0;
  let lpR = 0;
  const drive = p.distortion_on > 0.5 ? 1 + p.distortion_drive * 12 : 1;
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    let env: number;
    if (t < HOLD) env = t < a ? t / a : Math.max(s, 1 - ((t - a) / d) * (1 - s));
    else {
      const held = HOLD < a ? HOLD / a : Math.max(s, 1 - ((HOLD - a) / d) * (1 - s));
      env = held * Math.exp(-(t - HOLD) / (rel / 4));
    }
    const fenv = t < a2 ? t / a2 : Math.max(s2, 1 - ((t - a2) / d2) * (1 - s2));
    let sl = 0;
    let sr = 0;
    for (let v = 0; v < voices; v++) {
      const spread = voices === 1 ? 0 : (v / (voices - 1)) * 2 - 1;
      phases[v] = (phases[v] + (f0 * (1 + spread * det)) / SR) % 1;
      const saw = 2 * phases[v] - 1;
      const pan = 0.5 + 0.5 * spread * p.osc_1_stereo_spread;
      sl += saw * (1 - pan);
      sr += saw * pan;
    }
    sl /= Math.sqrt(voices);
    sr /= Math.sqrt(voices);
    const fc = Math.min(18000, cutoffBase * Math.pow(2, (envSemis * fenv) / 12));
    const k = 1 - Math.exp((-2 * Math.PI * fc) / SR);
    lpL += k * (sl - lpL);
    lpR += k * (sr - lpR);
    const lvl = p.osc_1_level * p.osc_1_level * 0.5;
    L[i] = Math.tanh(lpL * lvl * env * drive) / Math.tanh(drive);
    R[i] = Math.tanh(lpR * lvl * env * drive) / Math.tanh(drive);
  }
  if (p.reverb_on > 0.5) {
    const mix = p.reverb_dry_wet;
    const delay = Math.floor(SR * (0.03 + 0.05 * p.reverb_size));
    const fb = 0.5 + 0.45 * p.reverb_decay_time;
    for (let i = delay; i < n; i++) {
      L[i] += mix * fb * L[i - delay] * 0.6;
      R[i] += mix * fb * R[i - delay] * 0.6;
    }
  }
  return [L, R];
}

function wavUrl([L, R]: Float32Array[]): string {
  const n = L.length;
  const buf = new ArrayBuffer(44 + n * 4);
  const dv = new DataView(buf);
  const str = (o: number, s: string) => [...s].forEach((c, i) => dv.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  dv.setUint32(4, 36 + n * 4, true);
  str(8, "WAVE");
  str(12, "fmt ");
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 2, true);
  dv.setUint32(24, SR, true);
  dv.setUint32(28, SR * 4, true);
  dv.setUint16(32, 4, true);
  dv.setUint16(34, 16, true);
  str(36, "data");
  dv.setUint32(40, n * 4, true);
  for (let i = 0; i < n; i++) {
    dv.setInt16(44 + i * 4, Math.max(-1, Math.min(1, L[i])) * 32767, true);
    dv.setInt16(46 + i * 4, Math.max(-1, Math.min(1, R[i])) * 32767, true);
  }
  return URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
}

function describe(p: Record<string, number>): Descriptors {
  const cutoff = p.filter_1_on > 0.5 ? p.filter_1_cutoff : 1;
  const bright = 6.5 + 4 * cutoff * (p.filter_1_on > 0.5 ? p.filter_1_mix : 1) + (p.filter_1_cutoff_at_env_2_peak - p.filter_1_cutoff) * 1.2;
  const round2 = (x: number) => Math.round(x * 100) / 100;
  return {
    centroid_oct: round2(bright),
    hf_db: round2(-40 + 30 * cutoff),
    loudness_db: round2(-30 + 20 * p.osc_1_level),
    attack_s: round2(envTime(p.env_1_attack)),
    sustain: round2(p.env_1_sustain),
    tail_s: round2(Math.min(3, envTime(p.env_1_release) + (p.reverb_on > 0.5 ? p.reverb_decay_time * 1.2 : 0))),
    width: round2((p.osc_1_unison_voices > 0 ? p.osc_1_stereo_spread * 0.4 : 0) + (p.chorus_on > 0.5 ? 0.25 : 0)),
  };
}

// ---- session ----

interface MockRev {
  rev: Revision;
  values: Record<string, number>;
  summary: string;
}

/** The five-shape table of generated patches (synth_rl/wavetable.py): sine, triangle, saw, square, pulse at
 * positions 0, 0.25, 0.5, 0.75, 1, 128 samples each. Amplitudes are not matched as the server's are. */
const SHAPES: Wavetable = (() => {
  const N = 128;
  const phase = Array.from({ length: N }, (_, k) => (2 * Math.PI * (k + 0.5)) / N);
  const waves = [
    phase.map((p) => -Math.sin(p)),
    phase.map((p) => (-2 / Math.PI) * Math.asin(Math.sin(p))),
    phase.map((p) => 1 - ((p / Math.PI + 1) % 2)), // falls from 0 to -1, jumps to 1 mid-cycle, as the init saw does
    phase.map((p) => -Math.sign(Math.sin(p))),
    phase.map((p) => (Math.cos(p - 1.5 * Math.PI) > Math.cos(Math.PI / 4) ? 1 : -1 / 3)),
  ];
  return { positions: [0, 0.25, 0.5, 0.75, 1], frames: waves.map((w) => w.map((x) => Math.round(x * 1000) / 1000)) };
})();

const stack: MockRev[] = [];
let cursor = -1;
let nextRev = 0;
let presetName = "procedural-7a1d5006";
let lastManual: { name: string; at: number } | null = null;

function makeRevision(values: Record<string, number>, source: RevisionSource, parent: number | null): Revision {
  const rev = nextRev++;
  const controls: Control[] = EDITABLE.map(([name, label]) => ({
    name,
    label,
    group: groupOf(name),
    kind: kindOf(name),
    value: Math.round(values[name] * 1000) / 1000,
    display: display(name, values[name]),
  }));
  return {
    rev,
    parent,
    source,
    controls,
    descriptors: describe(values),
    audio_url: wavUrl(render(values)),
    preset_name: presetName,
    wavetables: { osc_1: SHAPES, osc_2: SHAPES },
  };
}

/** As on the server: while the peak equals the cutoff, it follows the cutoff. */
function follow(cur: Record<string, number>, edits: Record<string, number>): Record<string, number> {
  const peak = "filter_1_cutoff_at_env_2_peak";
  if ("filter_1_cutoff" in edits && !(peak in edits) && Math.abs(cur[peak] - cur.filter_1_cutoff) < 1e-6)
    return { ...edits, [peak]: edits.filter_1_cutoff };
  return edits;
}

function push(values: Record<string, number>, source: RevisionSource, summary: string) {
  const parent = cursor >= 0 ? stack[cursor].rev.rev : null;
  stack.splice(cursor + 1);
  stack.push({ rev: makeRevision(values, source, parent), values: { ...values }, summary });
  cursor = stack.length - 1;
}

function snapshot(): State {
  return {
    current: stack[cursor].rev,
    can_undo: cursor > 0,
    can_redo: cursor < stack.length - 1,
    history: stack.map((r) => ({ rev: r.rev.rev, source: r.rev.source, summary: r.summary })),
  };
}

function reset(values: Record<string, number>, source: RevisionSource, summary: string) {
  stack.length = 0;
  cursor = -1;
  push(values, source, summary);
}

const wait = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      reject(new DOMException("aborted", "AbortError"));
    });
  });

function ruleEdits(
  text: string,
  v: Record<string, number>,
  round: number,
): { reason: string; edits: Record<string, number>; clipped: string[] } {
  const t = text.toLowerCase();
  const edits: Record<string, number> = {};
  const reasons: string[] = [];
  const clipped: string[] = [];
  const step = round === 1 ? 1 : 0.4;
  const bump = (name: string, by: number) => {
    const want = v[name] + by * step;
    if (want < 0 || want > 1) clipped.push(name);
    edits[name] = Math.round(Math.max(0, Math.min(1, want)) * 1000) / 1000;
  };
  if (t.includes("bright")) {
    bump("filter_1_cutoff", 0.25);
    reasons.push("the top end sounds dull, so the filter opens");
  }
  if (t.includes("dark")) {
    bump("filter_1_cutoff", -0.25);
    reasons.push("the filter closes to take off the top");
  }
  if (t.includes("sweep") || t.includes("wah")) {
    edits.filter_1_on = 1;
    edits.filter_1_cutoff_at_env_2_peak = Math.min(1, Math.round((v.filter_1_cutoff + 0.3 * step) * 1000) / 1000);
    reasons.push("envelope 2 opens the filter on each note");
  }
  if (t.includes("dirt")) {
    edits.distortion_on = 1;
    bump("distortion_drive", 0.3);
    reasons.push("distortion adds grit");
  }
  if (t.includes("shorter tail") || t.includes("plucky")) {
    bump("env_1_release", -0.2);
    if (t.includes("plucky")) {
      bump("env_1_sustain", -0.4);
      bump("env_1_decay", -0.15);
    }
    reasons.push("the release is shortened so the note stops sooner");
  }
  if (t.includes("wide") || t.includes("spacious")) {
    edits.chorus_on = 1;
    edits.reverb_on = 1;
    bump("reverb_dry_wet", 0.2);
    bump("osc_1_stereo_spread", 0.3);
    reasons.push("chorus and reverb widen the image");
  }
  if (round > 1) reasons.unshift("Closer now; a smaller step in the same direction");
  return { reason: reasons.length ? reasons.join("; ") + "." : "Nothing in the request maps to a control.", edits, clipped };
}

async function* chat(body: ChatRequest, signal: AbortSignal): AsyncIterable<ChatEvent> {
  if (body.from_rev !== undefined) {
    const idx = stack.findIndex((r) => r.rev.rev === body.from_rev);
    if (idx >= 0) cursor = idx;
  }
  const fromRev = stack[cursor].rev.rev;
  const turnId = `t${Math.floor(Math.random() * 1000)}`;
  yield { event: "start", data: { turn_id: turnId, agent: body.agent, rounds: body.rounds, mute: body.mute, from_rev: fromRev } };
  const startDesc = stack[cursor].rev.descriptors;
  for (let round = 1; round <= body.rounds; round++) {
    const cur = stack[cursor];
    yield { event: "listening", data: { round, rev: cur.rev.rev } };
    await wait(body.agent === "rule" ? 500 : 2200, signal);
    if (body.text.includes("fail")) {
      yield { event: "error", data: { message: "Mock: the model returned HTTP 502 (simulated)." } };
      return;
    }
    const rule = body.mute && round > 1 ? { reason: "No change.", edits: {}, clipped: [] } : ruleEdits(body.text, cur.values, round);
    const { reason, clipped } = rule;
    const edits = follow(cur.values, rule.edits);
    const changes: Change[] = Object.entries(edits)
      .filter(([n, to]) => Math.abs(to - cur.values[n]) > 1e-6)
      .map(([name, to]) => ({
        name,
        from: cur.values[name],
        to,
        from_display: display(name, cur.values[name]),
        to_display: display(name, to),
      }));
    yield { event: "proposal", data: { round, reason, changes, clipped, secs: body.agent === "rule" ? 0.01 : 2.2 } };
    if (!changes.length) break;
    const values = { ...cur.values, ...edits };
    push(values, "agent", `${body.text} (round ${round})`);
    const after = stack[cursor].rev.descriptors;
    const delta: Delta = {};
    for (const k of DESCRIPTOR_KEYS) delta[k] = Math.round((after[k] - cur.rev.descriptors[k]) * 100) / 100;
    yield { event: "revision", data: { round, state: snapshot(), delta } };
  }
  const end = stack[cursor].rev.descriptors;
  const total: Delta = {};
  for (const k of DESCRIPTOR_KEYS) total[k] = Math.round((end[k] - startDesc[k]) * 100) / 100;
  yield { event: "done", data: { turn_id: turnId, from_rev: fromRev, to_rev: stack[cursor].rev.rev, delta: total } };
}

const AGENTS: AgentInfo[] = [
  { id: "openrouter:qwen/qwen3.8-omni-flash", label: "Qwen3.8-Omni-Flash (OpenRouter)", listens: true, available: true, why: "" },
  { id: "rule", label: "Keyword rule (never listens)", listens: false, available: true, why: "" },
  { id: "local:llama-server", label: "Local llama-server", listens: true, available: false, why: "not running (needs 64 GB; start it yourself)" },
];

export function createMockApi(): Api {
  reset(randomPatch(7), "new_patch", "new patch");
  return {
    state: async () => (await wait(150), snapshot()),
    agents: async () => AGENTS,
    newPatch: async (b) => {
      await wait(250);
      if (b.init) {
        presetName = "init";
        reset({ ...INIT }, "init", "init patch");
      } else {
        const seed = b.seed ?? Math.floor(Math.random() * 1e6);
        presetName = `procedural-${(seed * 2654435761 >>> 0).toString(16).padStart(8, "0").slice(0, 8)}`;
        reset(randomPatch(seed), "new_patch", "new patch");
      }
      return snapshot();
    },
    set: async (edits) => {
      await wait(120);
      const names = Object.keys(edits);
      const cur = stack[cursor];
      const values = { ...cur.values, ...follow(cur.values, edits) };
      const now = Date.now();
      if (names.length === 1 && cur.rev.source === "manual" && lastManual?.name === names[0] && now - lastManual.at < 2000 && cursor > 0) {
        cursor -= 1;
      }
      push(values, "manual", `set ${names.join(", ")}`);
      lastManual = names.length === 1 ? { name: names[0], at: now } : null;
      return snapshot();
    },
    undo: async () => {
      if (cursor > 0) cursor -= 1;
      return snapshot();
    },
    redo: async () => {
      if (cursor < stack.length - 1) cursor += 1;
      return snapshot();
    },
    openInVital: async () => ({ ok: true }),
    audioUrl: (rev) => rev.audio_url,
    presetUrl: () => "#",
    chat,
  };
}
