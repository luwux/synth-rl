// Where each control sits on the panel. Names, groups and kinds come from the server; this file only
// decides the arrangement, the short Vital-style label under each knob, and each section's colour.

import type { Group } from "../api/types";

/** Vital's size ratio: the panel is drawn at 1.2x the default skin's pixel sizes. */
export const S = 1.2;

export interface SectionSkin {
  /** Rotary Arc / Power Button On of this section's skin override */
  accent: string;
  /** Body Heading Background; the Filter override makes it transparent */
  heading: string;
}

const DEFAULT_ACCENT = "#aa88ff";
export const SKINS: Record<Group, SectionSkin> = {
  osc1: { accent: DEFAULT_ACCENT, heading: "var(--color-heading)" },
  osc2: { accent: DEFAULT_ACCENT, heading: "var(--color-heading)" },
  filter: { accent: "#ffb74d", heading: "transparent" },
  env1: { accent: "#64ffda", heading: "var(--color-heading)" },
  env2: { accent: "#64ffda", heading: "var(--color-heading)" },
  chorus: { accent: DEFAULT_ACCENT, heading: "var(--color-heading)" },
  delay: { accent: "#ff99e9", heading: "var(--color-heading)" },
  distortion: { accent: "#ff5252", heading: "var(--color-heading)" },
  eq: { accent: "#fff6e1", heading: "var(--color-heading)" },
  reverb: { accent: "#8fa0ff", heading: "var(--color-heading)" },
};

export interface SectionDef {
  group: Group;
  title: string;
  /** Vital's default: the title runs up a strip on the left (SynthSection::paintSidewaysHeadingText).
   * Used for the oscillators; the short, wide sections here (filter, envelopes, effects) use a band on top,
   * as Vital's filters do. */
  sideways?: boolean;
  /** name of the `_on` switch shown as the power button, if the section has one */
  power?: string;
  /** knob-like controls, left to right */
  knobs: string[];
  /** draw an ADSR display from these names */
  envelope?: { attack: string; decay: string; sustain: string; release?: string };
  /** draw the oscillator's wave at the `frame` knob's position from `state.current.wavetables[table]`, left of the knobs */
  wave?: { frame: string; table: string };
  /** draw a filter response from these names; `peak` is the cutoff at envelope 2's peak */
  filter?: { cutoff: string; peak: string; resonance: string; drive: string; mix: string };
}

// Read left to right, top to bottom: OSC 1, OSC 2, FILTER 1, ENV 1, ENV 2, then the effects, as in Vital's
// voice page (oscillators above the filter) followed by its effects page.
export const ROWS: SectionDef[][] = [
  [
    {
      group: "osc1",
      title: "OSC 1",
      sideways: true,
      knobs: ["osc_1_level", "osc_1_wave_frame", "osc_1_unison_voices", "osc_1_unison_detune", "osc_1_stereo_spread"],
      wave: { frame: "osc_1_wave_frame", table: "osc_1" },
    },
    {
      group: "osc2",
      title: "OSC 2",
      sideways: true,
      power: "osc_2_on",
      knobs: ["osc_2_level", "osc_2_wave_frame", "osc_2_unison_voices", "osc_2_unison_detune"],
      wave: { frame: "osc_2_wave_frame", table: "osc_2" },
    },
  ],
  [
    {
      group: "filter",
      title: "FILTER 1",
      power: "filter_1_on",
      knobs: ["filter_1_cutoff", "filter_1_cutoff_at_env_2_peak", "filter_1_resonance", "filter_1_drive", "filter_1_mix"],
      filter: {
        cutoff: "filter_1_cutoff",
        peak: "filter_1_cutoff_at_env_2_peak",
        resonance: "filter_1_resonance", drive: "filter_1_drive",
        mix: "filter_1_mix",
      },
    },
    {
      group: "env1",
      title: "ENV 1",
      knobs: ["env_1_attack", "env_1_decay", "env_1_sustain", "env_1_release"],
      envelope: { attack: "env_1_attack", decay: "env_1_decay", sustain: "env_1_sustain", release: "env_1_release" },
    },
    {
      group: "env2",
      title: "ENV 2",
      knobs: ["env_2_attack", "env_2_decay", "env_2_sustain"],
      envelope: { attack: "env_2_attack", decay: "env_2_decay", sustain: "env_2_sustain" },
    },
  ],
  [
    { group: "chorus", title: "CHORUS", power: "chorus_on", knobs: ["chorus_dry_wet"] },
    { group: "delay", title: "DELAY", power: "delay_on", knobs: ["delay_dry_wet"] },
    { group: "distortion", title: "DISTORTION", power: "distortion_on", knobs: ["distortion_drive", "distortion_mix"] },
    { group: "eq", title: "EQ", power: "eq_on", knobs: ["eq_low_gain", "eq_high_gain"] },
    { group: "reverb", title: "REVERB", power: "reverb_on", knobs: ["reverb_dry_wet", "reverb_decay_time", "reverb_size"] },
  ],
];

/** Short label in the pill under each knob (Vital prints these in capitals). */
export const SHORT: Record<string, string> = {
  osc_1_level: "Level",
  osc_1_wave_frame: "Frame",
  osc_1_unison_voices: "Voices",
  osc_1_unison_detune: "Detune",
  osc_1_stereo_spread: "Stereo",
  osc_2_level: "Level",
  osc_2_wave_frame: "Frame",
  osc_2_unison_voices: "Voices",
  osc_2_unison_detune: "Detune",
  filter_1_cutoff: "Cutoff",
  filter_1_cutoff_at_env_2_peak: "Env Peak",
  filter_1_resonance: "Res",
  filter_1_drive: "Drive",
  filter_1_mix: "Mix",
  env_1_attack: "Attack",
  env_1_decay: "Decay",
  env_1_sustain: "Sustain",
  env_1_release: "Release",
  env_2_attack: "Attack",
  env_2_decay: "Decay",
  env_2_sustain: "Sustain",
  chorus_dry_wet: "Mix",
  delay_dry_wet: "Mix",
  distortion_drive: "Drive",
  distortion_mix: "Mix",
  eq_low_gain: "Low",
  eq_high_gain: "High",
  reverb_dry_wet: "Mix",
  reverb_decay_time: "Decay",
  reverb_size: "Size",
};

/** Knobs whose neutral point is the middle (Vital draws these arcs from 12 o'clock). */
export const BIPOLAR = new Set(["eq_low_gain", "eq_high_gain"]);

/** Vital's unison voice counts, 1 to 16. The contract only says "stepped". */
export const STEPS: Record<string, { count: number; first: number }> = {
  osc_1_unison_voices: { count: 16, first: 1 },
  osc_2_unison_voices: { count: 16, first: 1 },
};

export const ALL_NAMES = ROWS.flat().flatMap((s) => [...(s.power ? [s.power] : []), ...s.knobs]);
