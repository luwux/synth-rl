import { useEffect, type ReactNode } from "react";
import { Download, ExternalLink, LoaderCircle, Redo2, Shuffle, Undo2, RotateCcw } from "lucide-react";
import { toast } from "sonner";
import { api, isMock } from "../api";
import { useStore } from "../state/store";
import { isTypingTarget } from "../lib/hooks";

const SOURCE_TEXT: Record<string, string> = {
  init: "init patch",
  new_patch: "new patch",
  manual: "manual edit",
  agent: "model edit",
};

export function Header() {
  const state = useStore((s) => s.state);
  const busy = useStore((s) => s.busy);
  const running = useStore((s) => s.active !== null);
  const conn = useStore((s) => s.conn);
  const { newPatch, undo, redo } = useStore.getState();
  const cur = state?.current;
  const locked = running || conn !== "ok" || busy !== null;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== "z" || isTypingTarget(e.target)) return;
      const s = useStore.getState();
      if (s.active || s.busy) return;
      e.preventDefault();
      void (e.shiftKey ? s.redo() : s.undo());
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const openInVital = async () => {
    if (!cur) return;
    try {
      await api.openInVital(cur.rev);
      toast.success("Sent to Vital", { description: `Revision ${cur.rev} opened as a preset.` });
    } catch (err) {
      toast.error("Could not open Vital", { description: err instanceof Error ? err.message : String(err) });
    }
  };

  return (
    <header className="flex h-[46px] shrink-0 items-center gap-4 px-[10px]">
      <div className="flex w-[218px] shrink-0 items-baseline gap-2">
        <span className="font-title text-[19px] font-light tracking-[0.02em] text-white">synth-rl</span>
      </div>

      <div className="flex h-[30px] min-w-0 flex-1 items-center gap-3 rounded-[4px] bg-selector px-3" data-testid="patch">
        <span className="truncate font-title text-[14px] font-light text-white">{cur?.preset_name ?? (conn === "loading" ? "Loading…" : "—")}</span>
        {cur && (
          <span className="shrink-0 text-[10.5px] tracking-[0.04em] text-icon uppercase">
            rev {cur.rev} · {SOURCE_TEXT[cur.source] ?? cur.source}
          </span>
        )}
        {isMock && <span className="ml-auto shrink-0 rounded-[6px] bg-orange/15 px-1.5 text-[10px] tracking-[0.05em] text-orange uppercase">mock data</span>}
      </div>

      <div className="flex shrink-0 items-center gap-[6px]">
        <HeaderButton primary onClick={() => void newPatch(false)} disabled={locked} busy={busy === "new"} icon={<Shuffle size={12} />}>
          New patch
        </HeaderButton>
        <HeaderButton onClick={() => void newPatch(true)} disabled={locked} busy={busy === "init"} icon={<RotateCcw size={12} />}>
          Init patch
        </HeaderButton>
        <span className="mx-1 h-[16px] w-px bg-white/10" />
        <HeaderButton onClick={() => void undo()} disabled={locked || !state?.can_undo} busy={busy === "undo"} icon={<Undo2 size={12} />} title="Undo (⌘Z)">
          Undo
        </HeaderButton>
        <HeaderButton onClick={() => void redo()} disabled={locked || !state?.can_redo} busy={busy === "redo"} icon={<Redo2 size={12} />} title="Redo (⇧⌘Z)">
          Redo
        </HeaderButton>
        <span className="mx-1 h-[16px] w-px bg-white/10" />
        <a
          href={cur ? api.presetUrl(cur.rev) : undefined}
          download={cur ? `${cur.preset_name || "synth-rl"}-rev${cur.rev}.vital` : undefined}
          aria-disabled={!cur}
          className={`pressable flex h-[22px] items-center gap-1.5 rounded-[11px] bg-field px-2.5 text-[11.5px] text-icon-hover hover:bg-heading hover:text-white ${!cur ? "pointer-events-none opacity-40" : ""}`}
        >
          <Download size={12} />
          Download .vital
        </a>
        <HeaderButton onClick={() => void openInVital()} disabled={!cur || conn !== "ok"} icon={<ExternalLink size={12} />}>
          Open in Vital
        </HeaderButton>
      </div>
    </header>
  );
}

function HeaderButton({
  children,
  onClick,
  disabled,
  busy,
  primary,
  icon,
  title,
}: {
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  busy?: boolean;
  primary?: boolean;
  icon?: ReactNode;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`pressable flex h-[22px] items-center gap-1.5 rounded-[11px] px-2.5 text-[11.5px] disabled:opacity-40 ${
        primary
          ? "bg-purple font-bold text-ui-text hover:bg-purple-hover active:bg-purple-press"
          : "bg-field text-icon-hover hover:bg-heading hover:text-white"
      }`}
    >
      <span className={busy ? "spin inline-flex" : "inline-flex"}>{busy ? <LoaderCircle size={12} /> : icon}</span>
      {children}
    </button>
  );
}
