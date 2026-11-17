import { ComposerPrimitive, MessagePrimitive, ThreadPrimitive } from "@assistant-ui/react";
import { Select, Switch, ToggleGroup } from "radix-ui";
import { ArrowUp, Check, ChevronDown, Ear, EarOff, Square } from "lucide-react";
import { useStore } from "../state/store";
import { PendingCard, RoundCard, SummaryCard } from "./Cards";

const EXAMPLES = ["brighter, with a shorter tail", "wider and more spacious", "more plucky", "darker and dirtier"];

export function Thread() {
  return (
    <ThreadPrimitive.Root className="flex h-full min-h-0 flex-col">
      <ThreadPrimitive.Viewport className="vital-scroll flex min-h-0 flex-1 flex-col overflow-y-auto px-[10px] pt-[10px]">
        <ThreadPrimitive.Empty>
          <EmptyState />
        </ThreadPrimitive.Empty>
        <ThreadPrimitive.Messages>
          {({ message }) => (message.role === "user" ? <UserMessage /> : <AssistantMessage />)}
        </ThreadPrimitive.Messages>
        <div className="min-h-[10px] flex-1" />
      </ThreadPrimitive.Viewport>
      <Composer />
    </ThreadPrimitive.Root>
  );
}

function UserMessage() {
  return (
    <MessagePrimitive.Root className="enter-up mb-[8px] flex justify-end pl-10">
      <div className="rounded-[9px] rounded-br-[3px] bg-body px-[11px] py-[6px] text-[13px] leading-[18px] text-white shadow-[0_1px_2px_rgba(0,0,0,0.25)]">
        <MessagePrimitive.Parts />
      </div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="mb-[14px] flex flex-col">
      <MessagePrimitive.Parts
        components={{
          data: { by_name: { round: RoundCard, summary: SummaryCard, pending: PendingCard } },
          Empty: () => null,
        }}
      />
    </MessagePrimitive.Root>
  );
}

function EmptyState() {
  const conn = useStore((s) => s.conn);
  const agent = useStore((s) => s.agents.find((a) => a.id === s.agentId));
  const count = useStore((s) => s.state?.current.controls.length);
  return (
    <div className="enter-up mt-auto mb-2 flex flex-col gap-2 rounded-[6px] bg-field px-[12px] py-[11px]">
      <div className="flex items-center gap-2 text-[10px] tracking-[0.07em] text-teal uppercase">
        <Ear size={12} /> How it works
      </div>
      <p className="text-[13px] leading-[19px] text-text">
        Describe the sound you want. The model hears the current patch (a held C4, 3 s) and sets some of the{" "}
        {count ? `${count} ` : ""}controls on the left. The knobs move, it hears the result, and it goes on step by step
        until it judges the sound right or reaches the step limit.
      </p>
      <p className="text-[12px] leading-[17px] text-white/45">
        {conn !== "ok"
          ? "Waiting for the synth server…"
          : agent
            ? agent.listens
              ? `${agent.label} listens to the audio.`
              : `${agent.label}: edits from the words alone, without listening.`
            : "No agent available."}{" "}
        Drag a knob to edit by hand (Shift for fine steps). Space plays the sound.
      </p>
    </div>
  );
}

function Composer() {
  const running = useStore((s) => s.active !== null);
  return (
    <div className="shrink-0 border-t border-black/30 bg-[#262a2e] px-[10px] pt-[8px] pb-[10px]">
      <div className="mb-[8px] flex flex-wrap gap-[5px]" data-testid="examples">
        {EXAMPLES.map((text) => (
          <ThreadPrimitive.Suggestion
            key={text}
            prompt={text}
            send
            className="pressable h-[22px] rounded-[11px] bg-field px-2.5 text-[11.5px] text-icon-hover shadow-[inset_0_0_0_1px_rgba(255,255,255,0.05)] hover:bg-heading hover:text-white disabled:opacity-40"
          >
            {text}
          </ThreadPrimitive.Suggestion>
        ))}
      </div>
      <ComposerPrimitive.Root className="flex items-end gap-2 rounded-[8px] bg-widget px-[10px] py-[7px] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)] focus-within:shadow-[inset_0_0_0_1px_rgba(29,233,182,0.45)]">
        <ComposerPrimitive.Input
          placeholder={running ? "The model is working…" : "How should the sound change?"}
          rows={1}
          className="max-h-[96px] min-h-[22px] flex-1 resize-none bg-transparent py-[2px] text-[13px] leading-[18px] text-white outline-none placeholder:text-white/30"
        />
        {running ? (
          <ComposerPrimitive.Cancel
            className="pressable flex h-[24px] w-[24px] shrink-0 items-center justify-center rounded-full bg-heading text-white hover:bg-body"
            aria-label="Stop"
          >
            <Square size={10} fill="currentColor" />
          </ComposerPrimitive.Cancel>
        ) : (
          <ComposerPrimitive.Send
            className="pressable flex h-[24px] w-[24px] shrink-0 items-center justify-center rounded-full bg-teal text-ui-text hover:bg-teal-light disabled:bg-heading disabled:text-white/30"
            aria-label="Send"
          >
            <ArrowUp size={14} strokeWidth={2.6} />
          </ComposerPrimitive.Send>
        )}
      </ComposerPrimitive.Root>
      <ComposerControls />
    </div>
  );
}

function ComposerControls() {
  const agents = useStore((s) => s.agents);
  const agentId = useStore((s) => s.agentId);
  const rounds = useStore((s) => s.rounds);
  const muted = useStore((s) => s.muted);
  const running = useStore((s) => s.active !== null);
  const { setAgent, setRounds, setMuted } = useStore.getState();
  const current = agents.find((a) => a.id === agentId);

  return (
    <div className="mt-[8px] flex items-center gap-[10px]">
      <Select.Root value={agentId} onValueChange={setAgent} disabled={running || agents.length === 0}>
        <Select.Trigger
          aria-label="Agent"
          data-testid="agent"
          className="flex h-[22px] min-w-0 flex-1 items-center gap-1.5 rounded-[4px] bg-field px-2 text-left text-[11.5px] text-text outline-none hover:bg-heading disabled:opacity-50"
        >
          {current?.listens ? <Ear size={12} className="shrink-0 text-teal" /> : <EarOff size={12} className="shrink-0 text-icon" />}
          <span className="min-w-0 flex-1 truncate">
            <Select.Value placeholder="No agent" />
          </span>
          <ChevronDown size={12} className="shrink-0 text-icon" />
        </Select.Trigger>
        <Select.Portal>
          <Select.Content
            position="popper"
            side="top"
            sideOffset={4}
            className="vital-select z-50 min-w-[var(--radix-select-trigger-width)] overflow-hidden rounded-[4px] border border-black bg-popup p-[3px] shadow-[0_8px_24px_rgba(0,0,0,0.5)]"
          >
            <Select.Viewport>
              {agents.map((a) => (
                <Select.Item
                  key={a.id}
                  value={a.id}
                  disabled={!a.available}
                  className="relative flex cursor-default flex-col rounded-[3px] py-[4px] pr-2 pl-6 text-[12px] text-text outline-none select-none data-[disabled]:opacity-45 data-[highlighted]:bg-[#906de9] data-[highlighted]:text-white"
                >
                  <Select.ItemIndicator className="absolute top-[6px] left-[6px]">
                    <Check size={12} />
                  </Select.ItemIndicator>
                  <Select.ItemText>{a.label}</Select.ItemText>
                  <span className="text-[10.5px] text-white/45">
                    {a.available ? (a.listens ? "listens to the audio" : "never listens") : a.why || "not available"}
                  </span>
                </Select.Item>
              ))}
            </Select.Viewport>
          </Select.Content>
        </Select.Portal>
      </Select.Root>

      <div className="flex items-center gap-1.5">
        <span className="text-[10px] tracking-[0.06em] text-icon uppercase">Max steps</span>
        <ToggleGroup.Root
          type="single"
          value={String(rounds)}
          onValueChange={(v) => v && setRounds(Number(v))}
          disabled={running}
          aria-label="Max steps"
          className="flex h-[22px] rounded-[4px] bg-field p-[2px]"
        >
          {[1, 2, 4, 6].map((n) => (
            <ToggleGroup.Item
              key={n}
              value={String(n)}
              className="w-[20px] rounded-[3px] text-[11.5px] tabular-nums text-icon transition-colors duration-150 hover:text-icon-hover disabled:opacity-50 data-[state=on]:bg-heading data-[state=on]:text-white"
            >
              {n}
            </ToggleGroup.Item>
          ))}
        </ToggleGroup.Root>
      </div>

      <label className="flex items-center gap-1.5 text-[10px] tracking-[0.06em] text-icon uppercase" title="Send silent audio instead of the sound, same prompt">
        <Switch.Root
          checked={muted}
          onCheckedChange={setMuted}
          disabled={running}
          data-testid="muted"
          className="relative h-[14px] w-[26px] rounded-full bg-field shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)] transition-colors duration-150 disabled:opacity-50 data-[state=checked]:bg-[color-mix(in_srgb,var(--color-orange)_35%,var(--color-field))]"
        >
          <Switch.Thumb className="block h-[10px] w-[10px] translate-x-[2px] rounded-full bg-power-off transition-[transform,background-color] duration-150 ease-[var(--ease-out)] data-[state=checked]:translate-x-[14px] data-[state=checked]:bg-orange" />
        </Switch.Root>
        Muted
      </label>
    </div>
  );
}
