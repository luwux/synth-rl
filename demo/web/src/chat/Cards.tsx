import { useEffect, useState } from "react";
import { AlertTriangle, Ear, EarOff, GitCompareArrows, Play, Square } from "lucide-react";
import type { DataMessagePartProps } from "@assistant-ui/react";
import type { Change } from "../api/types";
import { useStore, type RoundView, type Run } from "../state/store";
import { SKINS } from "../panel/layout";
import { prettyDisplay } from "../lib/format";

type Ids = { turnId: string; round?: number };

const accentOf = (name: string) => {
  const c = useStore.getState().state?.current.controls.find((x) => x.name === name);
  return c ? SKINS[c.group].accent : "#aa88ff";
};
const labelOf = (name: string) => useStore.getState().state?.current.controls.find((c) => c.name === name)?.label ?? name.replace(/_/g, " ");

function Spinner({ className = "" }: { className?: string }) {
  return (
    <span
      className={`spin inline-block h-[11px] w-[11px] rounded-full border-[1.6px] border-teal/25 border-t-teal ${className}`}
      aria-hidden
    />
  );
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 100);
    return () => window.clearInterval(t);
  }, []);
  return <span className="tabular-nums">{((now - since) / 1000).toFixed(1)} s</span>;
}

/** A level drop this large in one round usually means the edit removed the sound rather than shaped it. */
const QUIETER_WARNING_DB = -12;

function ChangeRows({ changes, dense = false }: { changes: Change[]; dense?: boolean }) {
  const flash = useStore((s) => s.flashControl);
  return (
    <ul className="flex flex-col">
      {changes.map((c, i) => (
        <li key={c.name} className="enter-up" style={{ animationDelay: `${60 + i * 40}ms` }}>
          <button
            type="button"
            onClick={() => flash(c.name)}
            data-change={c.name}
            title="Show this control on the panel"
            className={`group flex w-full items-center gap-2 rounded-[5px] px-1.5 text-left transition-colors duration-150 hover:bg-white/[0.06] ${dense ? "h-[20px]" : "h-[22px]"}`}
          >
            <span className="h-[6px] w-[6px] shrink-0 rounded-full" style={{ background: accentOf(c.name) }} />
            <span className="min-w-0 flex-1 truncate text-[12px] text-text">{labelOf(c.name)}</span>
            <span className="shrink-0 text-[12px] tabular-nums text-white/45">{prettyDisplay(c.from_display) || c.from.toFixed(2)}</span>
            <span className="shrink-0 text-[11px] text-white/35">→</span>
            <span className="shrink-0 text-[12px] tabular-nums text-white">{prettyDisplay(c.to_display) || c.to.toFixed(2)}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}

function RoundBody({ r, listensLabel }: { r: RoundView; listensLabel: string }) {
  if (r.status === "failed") {
    return (
      <div className="flex items-center gap-2 py-[2px] text-[12.5px] text-white/50">
        <AlertTriangle size={12} className="text-red/80" /> No result for this step.
      </div>
    );
  }
  if (r.status === "listening") {
    return (
      <div className="flex items-center gap-2 py-[2px] text-[12.5px] text-text">
        <Spinner />
        <span>
          {listensLabel} <span className="text-white/45">rev {r.listenRev}</span>
        </span>
        <span className="ml-auto text-[11px] text-white/40">
          <Elapsed since={r.startedAt} />
        </span>
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-[6px]">
      {r.reason && <p className="enter-up text-[13px] leading-[18px] text-text">{r.reason}</p>}
      {r.changes.length > 0 ? (
        <ChangeRows changes={r.changes} />
      ) : (
        <p className="text-[12px] text-white/45" data-stop={r.done ? "done" : "nochange"}>
          {r.done
            ? r.round === 1
              ? "No change to the sound."
              : "Done: it listened to the result and stopped here."
            : "No changes proposed, so the turn ends here."}
        </p>
      )}
      {r.clipped && r.clipped.length > 0 && (
        <div className="enter-up flex items-start gap-2 text-[12px] text-orange/90" data-warning="clipped">
          <AlertTriangle size={12} className="mt-[3px] shrink-0" />
          <span className="min-w-0">Clipped to the allowed range: {r.clipped.map(labelOf).join(", ")}</span>
        </div>
      )}
      {r.status === "applied" && r.sameAudio && (
        <div className="enter-up flex items-center gap-2 text-[12px] text-orange/90" data-warning="same-audio">
          <AlertTriangle size={12} className="shrink-0" />
          These changes made no difference to the sound: the new render is identical.
        </div>
      )}
      {r.status === "applied" && (r.delta?.loudness_db ?? 0) <= QUIETER_WARNING_DB && (
        <div className="enter-up flex items-center gap-2 text-[12px] text-red/90" data-warning="quieter">
          <AlertTriangle size={12} className="shrink-0" />
          This step made the sound {Math.abs(r.delta?.loudness_db ?? 0).toFixed(1)} dB quieter.
        </div>
      )}
      {r.status === "proposed" && (
        <div className="flex items-center gap-2 text-[11px] text-white/45">
          <Spinner /> rendering the new sound…
        </div>
      )}
    </div>
  );
}

function useRun(turnId: string, which: "main" | "compare" = "main") {
  return useStore((s) => s.turns.find((t) => t.id === turnId)?.[which]);
}

function listensText(run: Run) {
  const agent = useStore.getState().agents.find((a) => a.id === run.agent);
  if (run.mute) return "Sending silent audio of";
  if (agent && !agent.listens) return "Reading the request against";
  return "Listening to";
}

export function RoundCard({ data }: DataMessagePartProps<Ids>) {
  const run = useRun(data.turnId);
  const r = run?.rounds.find((x) => x.round === data.round);
  if (!run || !r) return null;
  return (
    <div className="enter-up mb-[6px] rounded-[6px] bg-field px-[10px] pt-[7px] pb-[9px] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.03)]" data-round={r.round}>
      <div className="mb-[5px] flex items-center gap-2 text-[10px] tracking-[0.07em] text-white/40 uppercase">
        <span className={r.status === "listening" ? "text-teal" : r.status === "failed" ? "text-red/80" : "text-white/55"}>
          Step {r.round}
          <span className="text-white/30"> / up to {run.roundsWanted}</span>
        </span>
        {run.mute && (
          <span className="flex items-center gap-1 text-orange/80">
            <EarOff size={10} /> muted
          </span>
        )}
        {r.secs !== undefined && <span className="ml-auto tabular-nums normal-case tracking-normal">{r.secs.toFixed(1)} s</span>}
      </div>
      <RoundBody r={r} listensLabel={listensText(run)} />
    </div>
  );
}

export function PendingCard({ data }: DataMessagePartProps<Ids>) {
  const run = useRun(data.turnId);
  if (!run || run.status !== "running") return null;
  return (
    <div className="enter-up mb-[6px] flex items-center gap-2 rounded-[6px] bg-field px-[10px] py-[9px] text-[12.5px] text-text">
      <Spinner /> Starting the turn…
    </div>
  );
}

function RunOutcome({ run }: { run: Run }) {
  if (run.status === "error")
    return (
      <div className="flex items-start gap-2 rounded-[6px] bg-red/10 px-[10px] py-[8px] text-[12.5px] text-[#ffb3b3] shadow-[inset_0_0_0_1px_rgba(255,82,82,0.25)]">
        <AlertTriangle size={14} className="mt-[2px] shrink-0 text-red" />
        <span className="min-w-0 [overflow-wrap:anywhere]">
          {run.error || "The turn failed."}
          <span className="block text-[11px] text-white/45">The panel shows whatever the server reached.</span>
        </span>
      </div>
    );
  if (run.status === "stopped")
    return (
      <div className="flex items-center gap-2 rounded-[6px] bg-field px-[10px] py-[7px] text-[12px] text-white/55">
        <Square size={11} /> Stopped. The panel shows the server’s current state.
      </div>
    );
  return null;
}

function countChanges(run: Run) {
  return new Set(run.rounds.flatMap((r) => (r.status === "applied" ? r.changes.map((c) => c.name) : []))).size;
}

function PlayRev({ rev, label }: { rev: number | undefined; label: string }) {
  const url = useStore((s) => (rev === undefined ? undefined : (s.revAudio[rev] ?? `/api/wav/${rev}.wav`)));
  const [playing, setPlaying] = useState(false);
  const [audio] = useState(() => new Audio());
  useEffect(() => () => audio.pause(), [audio]);
  if (rev === undefined || !url) return null;
  return (
    <button
      type="button"
      className="pressable flex h-[20px] items-center gap-1 rounded-[9px] bg-heading px-2 text-[10.5px] tracking-[0.04em] text-white/80 uppercase hover:bg-body"
      onClick={() => {
        if (playing) {
          audio.pause();
          setPlaying(false);
          return;
        }
        audio.src = url;
        audio.currentTime = 0;
        audio.onended = () => setPlaying(false);
        void audio.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
      }}
    >
      {playing ? <Square size={9} fill="currentColor" /> : <Play size={9} fill="currentColor" />}
      {label}
    </button>
  );
}

function CompareColumn({ title, run, icon }: { title: string; run: Run; icon: React.ReactNode }) {
  const changes = run.rounds.flatMap((r) => (r.status === "applied" ? r.changes : []));
  const last = new Map<string, Change>();
  for (const c of changes) {
    const prev = last.get(c.name);
    last.set(c.name, prev ? { ...c, from: prev.from, from_display: prev.from_display } : c);
  }
  const live = run.rounds.find((r) => r.status === "listening" || r.status === "proposed");
  return (
    <div className="flex min-w-0 flex-1 flex-col gap-[5px] rounded-[6px] bg-field px-[8px] pt-[7px] pb-[8px]">
      <div className="flex items-center gap-1.5 text-[10px] tracking-[0.07em] text-white/55 uppercase">
        {icon}
        {title}
      </div>
      {run.status === "running" && (
        <div className="flex items-center gap-2 text-[12px] text-text">
          <Spinner /> {live ? `Step ${live.round}: ${live.status === "listening" ? "waiting for the model" : "rendering"}` : "Starting…"}
        </div>
      )}
      {run.status !== "running" && run.status !== "done" && <RunOutcome run={run} />}
      {(run.status === "done" || changes.length > 0) && (
        <>
          <div className="text-[11.5px] text-white/60">
            {last.size} control{last.size === 1 ? "" : "s"} changed
            {run.toRev !== undefined && <span className="text-white/35"> · rev {run.toRev}</span>}
          </div>
          {last.size > 0 && <ChangeRows changes={[...last.values()]} dense />}
          <div className="mt-[2px]">
            <PlayRev rev={run.toRev} label="Play result" />
          </div>
        </>
      )}
    </div>
  );
}

export function SummaryCard({ data }: DataMessagePartProps<Ids>) {
  const turn = useStore((s) => s.turns.find((t) => t.id === data.turnId));
  const busy = useStore((s) => s.active !== null || s.conn !== "ok");
  const compare = useStore((s) => s.compare);
  if (!turn) return null;
  const { main } = turn;
  if (main.status === "running") return null;
  const outcome = <RunOutcome run={main} />;
  const n = countChanges(main);
  const applied = main.rounds.filter((r) => r.status === "applied").length;
  const compareLabel = main.mute ? "Compare with real audio" : "Compare with muted audio";

  return (
    <div className="enter-up flex flex-col gap-[6px]">
      {outcome}
      {main.status === "done" && !turn.compare && (
        <div className="flex items-center gap-2 px-[2px]">
          <span className="text-[11.5px] text-white/50">
            {applied} step{applied === 1 ? "" : "s"} · {n} control
            {n === 1 ? "" : "s"} changed
            {main.fromRev !== null && main.toRev !== undefined && (
              <span className="text-white/35">
                {" "}
                · rev {main.fromRev} → {main.toRev}
              </span>
            )}
          </span>
          <button
            type="button"
            disabled={busy}
            onClick={() => void compare(turn.id)}
            data-testid="compare"
            className="pressable ml-auto flex h-[22px] shrink-0 items-center gap-1.5 rounded-[11px] bg-heading px-2.5 text-[11.5px] text-white/85 hover:bg-body disabled:opacity-40"
          >
            <GitCompareArrows size={12} />
            {compareLabel}
          </button>
        </div>
      )}
      {turn.compare && (
        <div className="flex flex-col gap-[5px]" data-testid="comparison">
          <div className="flex items-center gap-2 px-[2px] text-[10px] tracking-[0.07em] text-white/40 uppercase">
            Same request from rev {main.fromRev}
          </div>
          <div className="flex gap-[6px]">
            <CompareColumn
              title={main.mute ? "Muted audio" : "With audio"}
              run={main}
              icon={main.mute ? <EarOff size={11} className="text-orange" /> : <Ear size={11} className="text-teal" />}
            />
            <CompareColumn
              title={turn.compare.mute ? "Muted audio" : "With audio"}
              run={turn.compare}
              icon={turn.compare.mute ? <EarOff size={11} className="text-orange" /> : <Ear size={11} className="text-teal" />}
            />
          </div>
        </div>
      )}
    </div>
  );
}
