import { useEffect } from "react";
import { Tooltip } from "radix-ui";
import { Toaster } from "sonner";
import { RefreshCw, Unplug } from "lucide-react";
import { useStore } from "./state/store";
import { Header } from "./ui/Header";
import { SynthPanel } from "./panel/SynthPanel";
import { SoundStrip } from "./sound/SoundStrip";
import { ChatRuntimeProvider } from "./chat/ChatRuntime";
import { Thread } from "./chat/Thread";
import { S } from "./panel/layout";

export function App() {
  const conn = useStore((s) => s.conn);
  useEffect(() => {
    void useStore.getState().init();
  }, []);

  return (
    <Tooltip.Provider delayDuration={0} skipDelayDuration={0}>
      <div className="flex h-full min-h-[760px] flex-col">
        <Header />
        <main className="flex min-h-0 flex-1 gap-[8px] px-[10px] pb-[10px]">
          <div className="relative flex min-w-0 basis-2/3 flex-col gap-[6px]" data-testid="synth">
            <SynthPanel />
            <div className="flex min-h-0 flex-1">
              <SoundStrip />
            </div>
            {conn !== "ok" && <ConnectionOverlay />}
          </div>
          <aside className="flex min-w-0 basis-1/3 flex-col overflow-hidden rounded-[4px] bg-widget shadow-[0_0_0_1px_var(--hairline),0_1px_3px_rgba(0,0,0,0.25)]">
            <header className="flex shrink-0 items-center justify-center bg-heading" style={{ height: 30 * S }}>
              <h2 className="font-light tracking-[0.06em] text-heading-text" style={{ fontSize: 15.5 }}>
                LISTENING MODEL
              </h2>
            </header>
            <div className="min-h-0 flex-1">
              <ChatRuntimeProvider>
                <Thread />
              </ChatRuntimeProvider>
            </div>
          </aside>
        </main>
      </div>
      <Toaster
        theme="dark"
        position="bottom-left"
        toastOptions={{
          style: {
            background: "var(--color-popup)",
            border: "1px solid #000",
            color: "var(--color-text)",
            fontFamily: "var(--font-sans)",
          },
        }}
      />
    </Tooltip.Provider>
  );
}

function ConnectionOverlay() {
  const conn = useStore((s) => s.conn);
  const err = useStore((s) => s.connError);
  if (conn === "loading")
    return (
      <div className="absolute inset-0 z-20 flex items-center justify-center rounded-[4px] bg-[rgba(38,42,45,0.55)]">
        <div className="flex items-center gap-2 rounded-[6px] bg-popup px-4 py-2.5 text-[12.5px] text-text shadow-lg">
          <span className="spin inline-block h-[12px] w-[12px] rounded-full border-2 border-purple/25 border-t-purple" />
          Loading the patch…
        </div>
      </div>
    );
  return (
    <div className="absolute inset-0 z-20 flex items-center justify-center rounded-[4px] bg-[rgba(29,33,37,0.78)] backdrop-blur-[2px]">
      <div className="enter-up flex w-[420px] flex-col gap-2 rounded-[6px] border border-black bg-popup px-5 py-4 shadow-[0_10px_30px_rgba(0,0,0,0.5)]">
        <div className="flex items-center gap-2 text-[13px] text-white">
          <Unplug size={15} className="text-red" /> The synth server is not answering
        </div>
        <p className="text-[12px] leading-[17px] text-white/55">{err || "No response from /api/state."}</p>
        <p className="text-[12px] leading-[17px] text-white/55">
          Start it from the repository root:
          <code className="mt-1 block rounded-[4px] bg-black/40 px-2 py-1 font-mono text-[11px] text-teal-light">
            .venv/bin/uvicorn demo.server.app:app --port 8765
          </code>
        </p>
        <button
          type="button"
          onClick={() => void useStore.getState().init()}
          className="pressable mt-1 flex h-[24px] w-fit items-center gap-1.5 rounded-[12px] bg-purple px-3 text-[12px] font-bold text-ui-text hover:bg-purple-hover"
        >
          <RefreshCw size={12} /> Try again
        </button>
      </div>
    </div>
  );
}
