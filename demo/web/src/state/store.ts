import { create } from "zustand";
import { toast } from "sonner";
import { api } from "../api";
import type { AgentInfo, Change, ChatRequest, Delta, State } from "../api/types";

export type RoundStatus = "listening" | "proposed" | "applied" | "nochange" | "failed";
export interface RoundView {
  round: number;
  status: RoundStatus;
  listenRev: number;
  reason?: string;
  changes: Change[];
  secs?: number;
  delta?: Delta;
  /** the round's render is byte for byte the audio it started from */
  sameAudio?: boolean;
  /** parameters the model asked to set outside their range, clipped by the server */
  clipped?: string[];
  /** the model said it is done */
  done?: boolean;
  rev?: number;
  startedAt: number;
}

export type RunStatus = "running" | "done" | "error" | "stopped";
export interface Run {
  agent: string;
  mute: boolean;
  roundsWanted: number;
  fromRev: number | null;
  rounds: RoundView[];
  status: RunStatus;
  error?: string;
  toRev?: number;
  delta?: Delta;
  startedAt: number;
}

export interface Turn {
  id: string;
  text: string;
  createdAt: Date;
  main: Run;
  compare?: Run;
}

export type Which = "main" | "compare";

interface Glow {
  /** increments for every agent round that changed something */
  seq: number;
  /** control name -> normalized value before that round */
  from: Record<string, number>;
}

interface Store {
  conn: "loading" | "ok" | "offline";
  connError: string;
  state: State | null;
  agents: AgentInfo[];
  agentId: string;
  rounds: number;
  muted: boolean;
  autoplay: boolean;
  turns: Turn[];
  active: { turnId: string; which: Which; abort: AbortController } | null;
  busy: string | null;
  glow: Glow;
  flash: { name: string; seq: number } | null;
  /** audio url of every revision seen, so "before this turn" can be played */
  revAudio: Record<number, string>;
  /** which revision the A/B switch compares against: the start of the latest run */
  before: { rev: number; url: string } | null;
  ab: "now" | "before";
  /** bumped to ask the sound strip to play the current audio */
  playSeq: number;
  /** manual value being dragged or sent, so the knob does not jump back while the request is in flight */
  pending: Record<string, number>;

  init(): Promise<void>;
  refresh(): Promise<void>;
  newPatch(init: boolean): Promise<void>;
  undo(): Promise<void>;
  redo(): Promise<void>;
  setControl(name: string, value: number): Promise<void>;
  setPending(name: string, value: number | null): void;
  send(text: string): Promise<void>;
  compare(turnId: string): Promise<void>;
  stop(): void;
  flashControl(name: string): void;
  setAgent(id: string): void;
  setRounds(n: number): void;
  setMuted(m: boolean): void;
  setAutoplay(a: boolean): void;
  setAb(ab: "now" | "before"): void;
}

const saved = (() => {
  try {
    return JSON.parse(localStorage.getItem("synth-rl-demo") ?? "{}") as Partial<Pick<Store, "agentId" | "rounds" | "autoplay">>;
  } catch {
    return {};
  }
})();
function persist(s: Pick<Store, "agentId" | "rounds" | "autoplay">) {
  try {
    localStorage.setItem("synth-rl-demo", JSON.stringify({ agentId: s.agentId, rounds: s.rounds, autoplay: s.autoplay }));
  } catch {
    /* private mode */
  }
}

const errorText = (err: unknown) => (err instanceof Error ? err.message : String(err));

let turnCounter = 0;

/** A round still waiting when its run ends did not produce a result. */
const failOpen = (r: RoundView): RoundView =>
  r.status === "listening" || r.status === "proposed" ? { ...r, status: "failed" } : r;

export const useStore = create<Store>((set, get) => {
  /** Put a new server state on screen. `agentChanges` marks the controls an agent round just moved. */
  const applyState = (state: State, agentChanges?: Change[]) => {
    const { glow, revAudio, pending } = get();
    const nextAudio = { ...revAudio, [state.current.rev]: state.current.audio_url };
    if (agentChanges && agentChanges.length) {
      const from: Record<string, number> = {};
      for (const c of agentChanges) from[c.name] = c.from;
      set({ state, revAudio: nextAudio, glow: { seq: glow.seq + 1, from }, pending: {} });
    } else {
      set({ state, revAudio: nextAudio, pending: Object.keys(pending).length ? {} : pending });
    }
  };
  const clearGlow = () => set((s) => ({ glow: { seq: s.glow.seq + 1, from: {} } }));

  const patchRun = (turnId: string, which: Which, fn: (run: Run) => Run) =>
    set((s) => ({
      turns: s.turns.map((t) => {
        if (t.id !== turnId) return t;
        const run = t[which];
        return run ? { ...t, [which]: fn(run) } : t;
      }),
    }));
  const patchRound = (turnId: string, which: Which, round: number, fn: (r: RoundView) => RoundView) =>
    patchRun(turnId, which, (run) => ({ ...run, rounds: run.rounds.map((r) => (r.round === round ? fn(r) : r)) }));

  const audioOf = (rev: number) => get().revAudio[rev] ?? `/api/wav/${rev}.wav`;

  async function runChat(turnId: string, which: Which, body: ChatRequest) {
    const abort = new AbortController();
    set({ active: { turnId, which, abort } });
    let finished = false;
    try {
      for await (const ev of api.chat(body, abort.signal)) {
        switch (ev.event) {
          case "start": {
            const fromRev = ev.data.from_rev;
            patchRun(turnId, which, (run) => ({ ...run, fromRev }));
            set({ before: { rev: fromRev, url: audioOf(fromRev) } });
            break;
          }
          case "listening": {
            const { round, rev } = ev.data;
            patchRun(turnId, which, (run) => ({
              ...run,
              rounds: [
                ...run.rounds.filter((r) => r.round !== round),
                { round, status: "listening", listenRev: rev, changes: [], startedAt: Date.now() },
              ],
            }));
            break;
          }
          case "proposal": {
            const { round, reason, changes, secs, clipped, done } = ev.data;
            patchRound(turnId, which, round, (r) => ({
              ...r,
              reason,
              changes,
              secs,
              clipped: clipped?.length ? clipped : undefined,
              done,
              status: changes.length ? "proposed" : "nochange",
            }));
            break;
          }
          case "revision": {
            const { round, state, delta, same_audio } = ev.data;
            const rev = ev.data.rev ?? state.current.rev;
            patchRound(turnId, which, round, (r) => ({ ...r, status: "applied", delta, sameAudio: same_audio, rev }));
            // A comparison is a side branch: the panel and the player stay on the current sound.
            if (which === "compare") break;
            const run = get().turns.find((t) => t.id === turnId)?.[which];
            const changes = run?.rounds.find((r) => r.round === round)?.changes ?? [];
            applyState(state, changes);
            set((s) => ({
              ab: "now",
              playSeq: s.autoplay ? s.playSeq + 1 : s.playSeq,
            }));
            break;
          }
          case "done": {
            finished = true;
            const { to_rev, delta } = ev.data;
            patchRun(turnId, which, (run) => ({ ...run, status: "done", toRev: to_rev, delta }));
            break;
          }
          case "error": {
            finished = true;
            patchRun(turnId, which, (run) => ({
              ...run,
              status: "error",
              error: ev.data.message,
              rounds: run.rounds.map(failOpen),
            }));
            break;
          }
        }
      }
      if (!finished) {
        patchRun(turnId, which, (run) => ({
          ...run,
          status: "error",
          error: "The connection closed before the turn finished.",
          rounds: run.rounds.map(failOpen),
        }));
      }
    } catch (err) {
      const aborted = abort.signal.aborted || (err instanceof DOMException && err.name === "AbortError");
      patchRun(turnId, which, (run) => ({
        ...run,
        status: aborted ? "stopped" : "error",
        error: aborted ? undefined : errorText(err),
        rounds: aborted ? run.rounds.filter((r) => r.status !== "listening") : run.rounds.map(failOpen),
      }));
    } finally {
      set({ active: null });
      if (!finished) await get().refresh();
    }
  }

  const newRun = (agent: string, mute: boolean, rounds: number, fromRev: number | null): Run => ({
    agent,
    mute,
    roundsWanted: rounds,
    fromRev,
    rounds: [],
    status: "running",
    startedAt: Date.now(),
  });

  return {
    conn: "loading",
    connError: "",
    state: null,
    agents: [],
    agentId: saved.agentId ?? "",
    rounds: [1, 2, 4, 6].includes(saved.rounds ?? 0) ? (saved.rounds as number) : 4,
    muted: false,
    autoplay: saved.autoplay ?? true,
    turns: [],
    active: null,
    busy: null,
    glow: { seq: 0, from: {} },
    flash: null,
    revAudio: {},
    before: null,
    ab: "now",
    playSeq: 0,
    pending: {},

    async init() {
      set({ conn: "loading" });
      try {
        const [state, agents] = await Promise.all([api.state(), api.agents()]);
        const { agentId } = get();
        const keep = agents.find((a) => a.id === agentId && a.available);
        const pick = keep ?? agents.find((a) => a.available && a.listens) ?? agents.find((a) => a.available) ?? agents[0];
        applyState(state);
        set({ conn: "ok", connError: "", agents, agentId: pick?.id ?? "" });
      } catch (err) {
        set({ conn: "offline", connError: errorText(err) });
      }
    },

    async refresh() {
      try {
        applyState(await api.state());
        if (get().conn !== "ok") set({ conn: "ok" });
      } catch (err) {
        set({ conn: "offline", connError: errorText(err) });
      }
    },

    async newPatch(init) {
      set({ busy: init ? "init" : "new" });
      try {
        applyState(await api.newPatch(init ? { init: true } : {}));
        clearGlow();
        set({ before: null, ab: "now" });
      } catch (err) {
        toast.error(init ? "Could not load the init patch" : "Could not make a new patch", { description: errorText(err) });
      } finally {
        set({ busy: null });
      }
    },

    async undo() {
      if (!get().state?.can_undo) return;
      set({ busy: "undo" });
      try {
        applyState(await api.undo());
        clearGlow();
      } catch (err) {
        toast.error("Undo failed", { description: errorText(err) });
      } finally {
        set({ busy: null });
      }
    },

    async redo() {
      if (!get().state?.can_redo) return;
      set({ busy: "redo" });
      try {
        applyState(await api.redo());
        clearGlow();
      } catch (err) {
        toast.error("Redo failed", { description: errorText(err) });
      } finally {
        set({ busy: null });
      }
    },

    setPending(name, value) {
      set((s) => {
        const pending = { ...s.pending };
        if (value === null) delete pending[name];
        else pending[name] = value;
        return { pending };
      });
    },

    async setControl(name, value) {
      get().setPending(name, value);
      try {
        const state = await api.set({ [name]: value });
        const { glow } = get();
        if (name in glow.from) {
          const from = { ...glow.from };
          delete from[name];
          set({ glow: { seq: glow.seq, from } });
        }
        applyState(state);
      } catch (err) {
        get().setPending(name, null);
        toast.error("The synth did not take that value", { description: errorText(err) });
      }
    },

    async send(text) {
      const { agentId, muted, rounds, active, state } = get();
      if (active || !text.trim()) return;
      const id = `turn-${++turnCounter}`;
      const turn: Turn = {
        id,
        text: text.trim(),
        createdAt: new Date(),
        main: newRun(agentId, muted, rounds, state?.current.rev ?? null),
      };
      set((s) => ({ turns: [...s.turns, turn] }));
      await runChat(id, "main", { text: turn.text, agent: agentId, rounds, mute: muted });
    },

    async compare(turnId) {
      const turn = get().turns.find((t) => t.id === turnId);
      if (!turn || get().active || turn.main.fromRev === null) return;
      const { main } = turn;
      // The comparison flips the audio: muted if the turn listened, listening if the turn was muted.
      const mute = !main.mute;
      set((s) => ({
        turns: s.turns.map((t) =>
          t.id === turnId ? { ...t, compare: newRun(main.agent, mute, main.roundsWanted, main.fromRev) } : t,
        ),
      }));
      await runChat(turnId, "compare", {
        text: turn.text,
        agent: main.agent,
        rounds: main.roundsWanted,
        mute,
        from_rev: main.fromRev!,
      });
    },

    stop() {
      get().active?.abort.abort();
    },

    flashControl(name) {
      set((s) => ({ flash: { name, seq: (s.flash?.seq ?? 0) + 1 } }));
    },

    setAgent(agentId) {
      set({ agentId });
      persist(get());
    },
    setRounds(rounds) {
      set({ rounds });
      persist(get());
    },
    setMuted(muted) {
      set({ muted });
    },
    setAutoplay(autoplay) {
      set({ autoplay });
      persist(get());
    },
    setAb(ab) {
      set({ ab });
    },
  };
});

export const controlOf = (state: State | null, name: string) => state?.current.controls.find((c) => c.name === name);
