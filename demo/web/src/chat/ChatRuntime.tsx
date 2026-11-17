import { useMemo, type ReactNode } from "react";
import { AssistantRuntimeProvider, useExternalStoreRuntime, type ThreadMessageLike } from "@assistant-ui/react";
import { useStore, type Turn } from "../state/store";

// assistant-ui renders the thread; the turns themselves live in the zustand store, which talks to /api/chat.
// Each assistant message is one data part per round ("data-round") plus a closing "data-summary" part.

function toMessages(turns: Turn[]): ThreadMessageLike[] {
  return turns.flatMap((t): ThreadMessageLike[] => {
    const running = t.main.status === "running" || t.compare?.status === "running";
    const parts: ThreadMessageLike["content"] = [
      ...(t.main.rounds.length === 0 && t.main.status === "running"
        ? [{ type: "data-pending" as const, data: { turnId: t.id } }]
        : []),
      ...t.main.rounds.map((r) => ({ type: "data-round" as const, id: `${t.id}-r${r.round}`, data: { turnId: t.id, round: r.round } })),
      { type: "data-summary" as const, data: { turnId: t.id } },
    ];
    return [
      { id: `${t.id}-u`, role: "user", content: [{ type: "text", text: t.text }], createdAt: t.createdAt },
      {
        id: `${t.id}-a`,
        role: "assistant",
        content: parts,
        createdAt: t.createdAt,
        status: running ? { type: "running" } : { type: "complete", reason: "stop" },
      },
    ];
  });
}

export function ChatRuntimeProvider({ children }: { children: ReactNode }) {
  const turns = useStore((s) => s.turns);
  const running = useStore((s) => s.active !== null);
  const disabled = useStore((s) => s.conn !== "ok" || !s.agentId);
  const messages = useMemo(() => toMessages(turns), [turns]);

  const runtime = useExternalStoreRuntime<ThreadMessageLike>({
    messages,
    isRunning: running,
    isDisabled: disabled,
    convertMessage: (m) => m,
    onNew: async (message) => {
      const text = message.content
        .map((p) => (p.type === "text" ? p.text : ""))
        .join("\n")
        .trim();
      if (text) await useStore.getState().send(text);
    },
    onCancel: async () => useStore.getState().stop(),
  });

  return <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>;
}
