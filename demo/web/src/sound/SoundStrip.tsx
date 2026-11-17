import { useEffect, useMemo, useRef, useState } from "react";
import { useWavesurfer } from "@wavesurfer/react";
import Spectrogram from "wavesurfer.js/plugins/spectrogram";
import { Switch, ToggleGroup } from "radix-ui";
import { Pause, Play } from "lucide-react";
import { useStore } from "../state/store";
import { isTypingTarget } from "../lib/hooks";
import { S } from "../panel/layout";

// 256-entry colour map in the panel's palette: widget background -> purple -> teal-white highlights.
function vitalColorMap(): number[][] {
  const stops: [number, [number, number, number, number]][] = [
    [0, [0x1d, 0x21, 0x25, 1]],
    [0.45, [0x2a, 0x26, 0x4a, 1]],
    [0.7, [0x6a, 0x50, 0xb8, 1]],
    [0.86, [0xaa, 0x88, 0xff, 1]],
    [0.96, [0x64, 0xff, 0xda, 1]],
    [1, [0xe8, 0xff, 0xf8, 1]],
  ];
  const out: number[][] = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, c0] = stops[k];
    const [t1, c1] = stops[k + 1];
    const u = (t - t0) / (t1 - t0 || 1);
    out.push(c0.map((c, j) => (j === 3 ? 1 : (c + (c1[j] - c) * u) / 255)));
  }
  return out;
}

const fmt = (s: number) => `${s.toFixed(2)} s`;
const WAVE_H = 64;

export function SoundStrip() {
  const state = useStore((s) => s.state);
  const before = useStore((s) => s.before);
  const ab = useStore((s) => s.ab);
  const setAb = useStore((s) => s.setAb);
  const autoplay = useStore((s) => s.autoplay);
  const setAutoplay = useStore((s) => s.setAutoplay);
  const playSeq = useStore((s) => s.playSeq);

  const url = ab === "before" && before ? before.url : state?.current.audio_url;
  const container = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const [boxH, setBoxH] = useState(0);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    let t: number | undefined;
    let first = true;
    const ro = new ResizeObserver(([e]) => {
      window.clearTimeout(t);
      const h = Math.round(e.contentRect.height);
      // The spectrogram's height is fixed when the plugin is made, so remake it only for real size changes.
      t = window.setTimeout(() => setBoxH((old) => (Math.abs(old - h) > 6 ? h : old)), first ? 0 : 250);
      first = false;
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      window.clearTimeout(t);
    };
  }, []);
  const specH = Math.max(60, boxH - WAVE_H - 8);
  const ready = boxH > 0;
  const plugins = useMemo(
    () =>
      ready
        ? [
            Spectrogram.create({
              height: specH,
              labels: false,
              fftSamples: 1024,
              frequencyMin: 30,
              frequencyMax: 16000,
              scale: "mel",
              colorMap: vitalColorMap(),
              gainDB: 22,
              rangeDB: 62,
            }),
          ]
        : [],
    [ready, specH],
  );
  const { wavesurfer, isPlaying, currentTime } = useWavesurfer({
    container,
    height: WAVE_H,
    sampleRate: 44100,
    waveColor: "rgba(170, 136, 255, 0.55)",
    progressColor: "#aa88ff",
    cursorColor: "rgba(255,255,255,0.8)",
    cursorWidth: 1,
    normalize: false,
    barHeight: 1.25,
    dragToSeek: true,
    plugins,
  });

  const [duration, setDuration] = useState(0);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const loading = useRef(false);
  const wantPlay = useRef(false);
  const resume = useRef<{ time: number; play: boolean } | null>(null);
  const loadedUrl = useRef<string | null>(null);
  const loadedWs = useRef<unknown>(null);

  useEffect(() => {
    if (!wavesurfer) return;
    const offs = [
      wavesurfer.on("ready", (d) => {
        loading.current = false;
        setDuration(d);
        setStatus("ready");
        const r = resume.current;
        resume.current = null;
        if (r) wavesurfer.setTime(Math.min(r.time, d));
        if (wantPlay.current || r?.play) {
          wantPlay.current = false;
          if (!r) wavesurfer.setTime(0);
          void wavesurfer.play().catch(() => {});
        }
      }),
      wavesurfer.on("error", () => {
        loading.current = false;
        setStatus("error");
      }),
    ];
    return () => offs.forEach((off) => off());
  }, [wavesurfer]);

  useEffect(() => {
    if (!wavesurfer || !url || (loadedUrl.current === url && loadedWs.current === wavesurfer)) return;
    const same = loadedWs.current === wavesurfer && loadedUrl.current !== null;
    // Switching A/B mid-playback keeps the position, like a hardware A/B switch.
    resume.current = same ? { time: wavesurfer.getCurrentTime(), play: wavesurfer.isPlaying() } : null;
    loadedUrl.current = url;
    loadedWs.current = wavesurfer;
    loading.current = true;
    setStatus("loading");
    wavesurfer.load(url).catch(() => {
      loading.current = false;
      setStatus("error");
    });
  }, [wavesurfer, url]);

  useEffect(() => {
    if (!wavesurfer || playSeq === 0) return;
    resume.current = null;
    if (loading.current) wantPlay.current = true;
    else {
      wavesurfer.setTime(0);
      void wavesurfer.play().catch(() => {});
    }
  }, [playSeq, wavesurfer]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" || isTypingTarget(e.target) || e.metaKey || e.ctrlKey) return;
      e.preventDefault();
      void wavesurfer?.playPause();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [wavesurfer]);

  return (
    <section className="flex min-w-0 flex-1 flex-col rounded-[4px] bg-body shadow-[0_0_0_1px_var(--hairline),0_1px_3px_rgba(0,0,0,0.25)]">
      <header className="relative flex shrink-0 items-center gap-2.5 rounded-t-[4px] bg-heading px-[8px]" style={{ height: 30 * S }}>
        <button
          type="button"
          onClick={() => void wavesurfer?.playPause()}
          disabled={status !== "ready"}
          aria-label={isPlaying ? "Pause" : "Play"}
          title="Play / pause (space)"
          data-testid="play"
          className="pressable flex h-[20px] items-center gap-1.5 rounded-[9px] bg-purple pr-2.5 pl-2 text-[11px] font-bold tracking-[0.04em] text-ui-text uppercase hover:bg-purple-hover active:bg-purple-press disabled:opacity-40"
        >
          {isPlaying ? <Pause size={11} strokeWidth={3} fill="currentColor" /> : <Play size={11} strokeWidth={3} fill="currentColor" />}
          {isPlaying ? "Pause" : "Play"}
        </button>
        <h2 className="font-light tracking-[0.06em] text-heading-text" style={{ fontSize: 15.5 }}>
          SOUND
        </h2>
        <span className="w-[86px] text-[11px] tabular-nums text-icon-hover">
          {fmt(currentTime)} / {fmt(duration)}
        </span>
        <div className="ml-auto flex items-center gap-3">
          <ToggleGroup.Root
            type="single"
            value={ab}
            onValueChange={(v) => v && setAb(v as "now" | "before")}
            disabled={!before}
            aria-label="Compare with the sound before this turn"
            className="flex h-[20px] overflow-hidden rounded-[9px] bg-field p-[2px]"
            title={before ? `A: rev ${before.rev}, before the last turn` : "Available after a turn"}
          >
            {(
              [
                ["before", "A  before turn"],
                ["now", "B  now"],
              ] as const
            ).map(([v, label]) => (
              <ToggleGroup.Item
                key={v}
                value={v}
                className="rounded-[7px] px-2 text-[10.5px] tracking-[0.04em] whitespace-pre text-icon uppercase transition-colors duration-150 hover:text-icon-hover disabled:opacity-40 data-[state=on]:bg-heading data-[state=on]:text-white"
              >
                {label}
              </ToggleGroup.Item>
            ))}
          </ToggleGroup.Root>
          <label className="flex items-center gap-1.5 text-[10.5px] tracking-[0.04em] text-icon-hover uppercase">
            <Switch.Root
              checked={autoplay}
              onCheckedChange={setAutoplay}
              className="relative h-[14px] w-[26px] rounded-full bg-field shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)] transition-colors duration-150 data-[state=checked]:bg-[color-mix(in_srgb,var(--color-purple)_35%,var(--color-field))]"
            >
              <Switch.Thumb className="block h-[10px] w-[10px] translate-x-[2px] rounded-full bg-power-off transition-[transform,background-color] duration-150 ease-[var(--ease-out)] data-[state=checked]:translate-x-[14px] data-[state=checked]:bg-purple" />
            </Switch.Root>
            Autoplay
          </label>
        </div>
      </header>
      <div ref={box} className="relative m-[6px] min-h-0 flex-1 overflow-hidden rounded-[6px] bg-widget">
        {ready && <div ref={container} className="wave relative pt-[4px]" />}
        {status !== "ready" && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-[11px] tracking-[0.06em] text-white/40 uppercase">
            {status === "error" ? "Audio could not be loaded" : !url ? "No sound yet" : "Rendering audio…"}
          </div>
        )}
        {ab === "before" && before && (
          <span className="pointer-events-none absolute top-[5px] left-[7px] rounded-[6px] bg-black/40 px-1.5 py-[1px] text-[10px] tracking-[0.05em] text-orange uppercase">
            A: before turn · rev {before.rev}
          </span>
        )}
      </div>
    </section>
  );
}
