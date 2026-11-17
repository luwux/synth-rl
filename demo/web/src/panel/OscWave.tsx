import { useId, useState } from "react";
import { useMotionValueEvent, type MotionValue } from "motion/react";
import type { Wavetable } from "../api/types";
import { useWidth } from "./useGlide";

// The oscillator's wavetable view, as in Vital's WavetableComponent: on Widget Background, the single cycle the
// oscillator plays at its current wave frame, a 2 px line in the section's accent with a bloom and a translucent
// fill to the centre line. It reads the FRAME knob's own displayed value, so it follows a drag, keyboard steps and
// the glide after an agent round. Between keyframes the wave is their straight blend, which is what Vital plays
// for a table that cross-fades in the time domain (synth_rl/wavetable.py).

interface Props {
  table: Wavetable | undefined;
  /** the FRAME knob's displayed value, 0-1 */
  frame: MotionValue<number>;
  accent: string;
  /** false when the oscillator is switched off: drawn dimmed */
  on: boolean;
  height: number;
  testId: string;
}

const PAD_X = 8;
const PAD_Y = 9;

/** The wave at knob value v: the straight blend of the keyframes around it; null if the table is unusable. */
export function waveAt(table: Wavetable | undefined, v: number): number[] | null {
  const pos = table?.positions;
  const frames = table?.frames;
  if (!pos || !frames || !pos.length || pos.length !== frames.length || !frames[0]?.length) return null;
  if (pos.length === 1 || v <= pos[0]) return frames[0];
  const last = pos.length - 1;
  if (v >= pos[last]) return frames[last];
  let i = 0;
  while (i < last - 1 && v > pos[i + 1]) i++;
  const span = pos[i + 1] - pos[i];
  const t = span > 1e-9 ? (v - pos[i]) / span : 0;
  const a = frames[i];
  const b = frames[i + 1];
  return a.map((x, k) => (1 - t) * x + t * (b[k] ?? x));
}

export function OscWave({ table, frame, accent, on, height, testId }: Props) {
  const gid = useId().replace(/:/g, "");
  const [ref, width] = useWidth<HTMLDivElement>(160);
  const [v, setV] = useState(frame.get());
  useMotionValueEvent(frame, "change", setV);

  const w = Math.max(width, 60);
  const h = height;
  const mid = h / 2;
  const amp = mid - PAD_Y;
  const wave = waveAt(table, v);

  let line = "";
  let fill = "";
  if (wave) {
    const n = wave.length;
    const xOf = (k: number) => PAD_X + ((k + 0.5) / n) * (w - 2 * PAD_X);
    const yOf = (y: number) => mid - Math.max(-1, Math.min(1, y)) * amp;
    line = wave.map((y, k) => `${k === 0 ? "M" : "L"} ${xOf(k).toFixed(2)} ${yOf(y).toFixed(2)}`).join(" ");
    fill = `${line} L ${xOf(n - 1).toFixed(2)} ${mid} L ${xOf(0).toFixed(2)} ${mid} Z`;
  }
  const stroke = on ? accent : "var(--color-arc-off)";
  const grid = [0.25, 0.5, 0.75].map((f) => PAD_X + f * (w - 2 * PAD_X));

  return (
    <div ref={ref} className="relative w-full overflow-hidden rounded-[6px] bg-widget" style={{ height }} data-testid={testId}>
      <svg width={w} height={h} className="block">
        <defs>
          <linearGradient id={`wfill-${gid}`} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor={stroke} stopOpacity={on ? 0.3 : 0.1} />
            <stop offset="0.5" stopColor={stroke} stopOpacity={on ? 0.06 : 0.02} />
            <stop offset="1" stopColor={stroke} stopOpacity={on ? 0.3 : 0.1} />
          </linearGradient>
        </defs>
        {grid.map((x) => (
          <line key={x} x1={x} x2={x} y1={0} y2={h} stroke="rgba(255,255,255,0.045)" strokeWidth={1} />
        ))}
        <line x1={0} x2={w} y1={mid} y2={mid} stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
        {wave && (
          <>
            <path d={fill} fill={`url(#wfill-${gid})`} />
            {on && <path d={line} fill="none" stroke={stroke} strokeWidth={4} opacity={0.5} style={{ filter: "blur(3px)" }} />}
            <path d={line} fill="none" stroke={stroke} strokeWidth={2} strokeLinejoin="round" opacity={on ? 1 : 0.6} />
          </>
        )}
      </svg>
      {!on && (
        <span className="pointer-events-none absolute top-[4px] right-[6px] rounded-[3px] bg-widget/85 px-[3px] text-[10px] tracking-[0.06em] text-white/30 uppercase">
          osc off
        </span>
      )}
    </div>
  );
}
