// Types from demo/DESIGN.md ("Server" > "Types (JSON)" and the /api/chat events). Keep in sync with the contract.

export type Group =
  | "osc1"
  | "osc2"
  | "filter"
  | "env1"
  | "env2"
  | "chorus"
  | "reverb"
  | "delay"
  | "distortion"
  | "eq";

export type ControlKind = "knob" | "switch" | "stepped";

export interface Control {
  name: string;
  label: string;
  group: Group;
  kind: ControlKind;
  /** normalized 0-1, exact */
  value: number;
  /** the value as text with its unit, from the server ("1240 Hz", "0.48 s", "3 voices", "on"); may be "" */
  display: string;
}

export const DESCRIPTOR_KEYS = [
  "centroid_oct",
  "hf_db",
  "loudness_db",
  "attack_s",
  "sustain",
  "tail_s",
  "width",
] as const;
export type DescriptorKey = (typeof DESCRIPTOR_KEYS)[number];
export type Descriptors = Record<DescriptorKey, number>;
export type Delta = Partial<Record<DescriptorKey, number>>;

export type RevisionSource = "init" | "new_patch" | "manual" | "agent";

export interface Revision {
  rev: number;
  parent: number | null;
  source: RevisionSource;
  controls: Control[];
  descriptors: Descriptors;
  audio_url: string;
  preset_name: string;
  /** per oscillator ("osc_1", "osc_2"), its wavetable's keyframes for the waveform display; absent on older servers */
  wavetables?: Record<string, Wavetable>;
}

export interface Wavetable {
  /** keyframe positions on the wave frame knob's 0-1 scale, ascending */
  positions: number[];
  /** one single-cycle wave per keyframe, samples in -1..1; the wave between two keyframes is their straight blend */
  frames: number[][];
}

export interface Change {
  name: string;
  from: number;
  to: number;
  from_display: string;
  to_display: string;
}

export interface HistoryEntry {
  rev: number;
  source: RevisionSource;
  summary: string;
}

export interface State {
  current: Revision;
  can_undo: boolean;
  can_redo: boolean;
  history: HistoryEntry[];
}

export interface AgentInfo {
  id: string;
  label: string;
  listens: boolean;
  available: boolean;
  why: string;
}

export interface ChatRequest {
  text: string;
  agent: string;
  rounds: number;
  mute: boolean;
  from_rev?: number;
}

export interface StartEvent {
  turn_id: string;
  agent: string;
  rounds: number;
  mute: boolean;
  from_rev: number;
}
export interface ListeningEvent {
  round: number;
  rev: number;
}
export interface ProposalEvent {
  round: number;
  reason: string;
  changes: Change[];
  secs: number;
  /** parameters the model asked to set outside their allowed range; the server clipped them to the range */
  clipped?: string[];
  /** the model said it is done (it heard the last result and stopped) */
  done?: boolean;
}
export interface RevisionEvent {
  round: number;
  /** the revision this step made; in a comparison branch it is not state.current */
  rev?: number;
  state: State;
  delta: Delta;
  /** the round's render is byte for byte the audio it started from */
  same_audio?: boolean;
}
export interface DoneEvent {
  turn_id: string;
  from_rev: number;
  to_rev: number;
  delta: Delta;
}
export interface ErrorEvent {
  message: string;
}

export type ChatEvent =
  | { event: "start"; data: StartEvent }
  | { event: "listening"; data: ListeningEvent }
  | { event: "proposal"; data: ProposalEvent }
  | { event: "revision"; data: RevisionEvent }
  | { event: "done"; data: DoneEvent }
  | { event: "error"; data: ErrorEvent };

export interface Api {
  state(): Promise<State>;
  agents(): Promise<AgentInfo[]>;
  newPatch(body: { seed?: number; init?: boolean }): Promise<State>;
  set(edits: Record<string, number>): Promise<State>;
  undo(): Promise<State>;
  redo(): Promise<State>;
  openInVital(rev: number): Promise<{ ok: boolean }>;
  audioUrl(rev: Revision): string;
  presetUrl(rev: number): string;
  chat(body: ChatRequest, signal: AbortSignal): AsyncIterable<ChatEvent>;
}
